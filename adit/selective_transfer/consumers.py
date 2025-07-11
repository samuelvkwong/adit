import asyncio
import contextlib
import logging
import threading
import zipfile
import secrets

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Iterator, Literal, cast, Callable
from pathlib import Path
from io import BytesIO
from functools import partial

from adit_radis_shared.accounts.models import User
from adit_radis_shared.common.utils.debounce import debounce
from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from crispy_forms.utils import render_crispy_form
from django.core.signing import TimestampSigner
from django.conf import settings
from pydicom import Dataset

from django.template.loader import render_to_string
from requests.exceptions import HTTPError

from adit.core.errors import DicomError, RetriableDicomError
from adit.core.models import DicomNode
from adit.core.utils.dicom_dataset import QueryDataset, ResultDataset
from adit.core.utils.dicom_operator import DicomOperator
from adit.core.utils.dicom_utils import write_dataset
from adit.core.utils.sanitize import sanitize_filename
from adit.core.utils.dicom_manipulator import DicomManipulator

from .forms import SelectiveTransferJobForm
from .models import SelectiveTransferJob, SelectiveTransferTask
from .views import SELECTIVE_TRANSFER_ADVANCED_OPTIONS_COLLAPSED

logger = logging.getLogger(__name__)

lock = threading.Lock()


@contextlib.asynccontextmanager
async def async_lock(_lock):
    """To handle a thread lock in async code.

    See https://stackoverflow.com/a/63425191/166229
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _lock.acquire)
    try:
        yield  # the lock is held
    finally:
        _lock.release()


def render_error_message(message) -> str:
    return render_to_string(
        "selective_transfer/_error_message.html",
        {"error_message": str(message)},
    )


class SelectiveTransferConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        logger.debug("Connected to WebSocket client.")

        self.user: User = self.scope["user"]
        self.query_operators: list[DicomOperator] = []
        self.current_message_id: int = 0
        self.pool = ThreadPoolExecutor()

        await self.accept()

    async def disconnect(self, code: int) -> None:
        await self._abort_operators()
        self.pool.shutdown(wait=False, cancel_futures=True)
        logger.debug("Disconnected from WebSocket client with code: %s", code)

    async def receive_json(self, content: dict[str, Any], **kwargs) -> None:
        error_response = await self.check_permission()
        if error_response:
            await self.send(error_response)
            return

        action: str = content.get("action", "")
        if action not in ["query", "cancel", "reset", "transfer", "direct_download"]:
            await self.send(render_error_message(f"Invalid action to process: {action}"))
            return

        async with async_lock(lock):
            # First we abort all operators as we received a new command what to do
            self.current_message_id += 1
            message_id = self.current_message_id
            await self._abort_operators()

        if action == "cancel" or action == "reset":
            # The connectors are already aborted, so we can just update the UI
            query_hint = render_to_string("selective_transfer/_query_hint.html")
            await self.send(query_hint)
            return

        # We are now in a query or transfer action so we have to process the form
        form = await self.get_form(action, content)
        form_valid: bool = await database_sync_to_async(form.is_valid)()

        if action == "query":
            if form_valid:
                in_progress_message = render_to_string("selective_transfer/_action_in_progress.html", {"loading_text": "Searching ..."})
                await self.send(in_progress_message)
                asyncio.create_task(self._make_query(form, message_id))
            else:
                form_error_response = await self._build_form_error_response(
                    form, "Please correct the form errors and search again."
                )
                await self.send(form_error_response)

        elif action == "transfer":
            if form_valid:
                asyncio.create_task(self.make_transfer(form))
            else:
                form_error_response = await self._build_form_error_response(
                    form, "Please correct the form errors and transfer again."
                )
                await self.send(form_error_response)
        
        elif action == "direct_download":
            logger.debug("Direct download action received.")
            if form_valid:
                in_progress_message = render_to_string("selective_transfer/_action_in_progress.html", {"loading_text": "Downloading ..."})
                await self.send(in_progress_message)
                asyncio.create_task(self._direct_download(form))
            else:
                form_error_response = await self._build_form_error_response(
                    form, "Please correct the form errors and download again."
                )
                await self.send(form_error_response)

        else:
            raise ValueError(f"Invalid action to process: {action}")

    @database_sync_to_async
    def get_form(
        self, action: Literal["query", "transfer"], content: dict["str", Any]
    ) -> SelectiveTransferJobForm:
        # Advanced options collapsed preference is not part of the form data itself,
        # so we have to pass it separately.
        advanced_options_collapsed = self.user.preferences.get(
            SELECTIVE_TRANSFER_ADVANCED_OPTIONS_COLLAPSED, False
        )

        # We are now in a query or transfer action so we have to process the form
        return SelectiveTransferJobForm(
            content,
            user=self.user,
            action=action,
            advanced_options_collapsed=advanced_options_collapsed,
        )

    async def _abort_operators(self) -> None:
        loop = asyncio.get_event_loop()
        while self.query_operators:
            for operator in self.query_operators[:]:
                self.query_operators.remove(operator)
                loop.call_soon_threadsafe(operator.abort)

    @database_sync_to_async
    def check_permission(self) -> str | None:
        if not self.user:
            return render_error_message("Access denied. You must be logged in.")

        if not self.user.has_perm("selective_transfer.add_selectivetransferjob"):
            return render_error_message("Access denied. You don't have the proper permission.")

        return None

    @database_sync_to_async
    def _build_form_error_response(self, form: SelectiveTransferJobForm, message: str) -> str:
        rendered_form: str = render_crispy_form(form)
        rendered_error_message: str = render_error_message(message)
        return rendered_form + rendered_error_message

    async def _make_query(self, form: SelectiveTransferJobForm, message_id: int) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self.pool, self._generate_and_send_query_response, form, message_id
            )
        except HTTPError as e:
            status_code = e.response.status_code
            additional_error_text = ""
            if status_code == 400:  # Bad request
                additional_error_text = "The source did not understand the ADIT request."
            elif status_code == 401 or status_code == 403:  # Unauthorized or forbidden
                additional_error_text = "ADIT is not authorized to access the source."
            elif status_code == 500:  # Internal server error
                additional_error_text = "The source could not process the ADIT request."
            elif status_code == 502:  # Bad gateway
                additional_error_text = "The source is not available."
            form_error_response = await self._build_form_error_response(
                form, f"Something went wrong at your requested source. {additional_error_text}"
            )
            await self.send(form_error_response)
        except (DicomError, RetriableDicomError):
            form_error_response = await self._build_form_error_response(
                form,
                "Something went wrong at your requested source. "
                "A Dicom Error has occured at the source.",
            )
            await self.send(form_error_response)
        except Exception:
            form_error_response = await self._build_form_error_response(
                form, "Something went wrong."
            )
            await self.send(form_error_response)

    def _generate_and_send_query_response(
        self, form: SelectiveTransferJobForm, message_id: int
    ) -> None:
        with lock:
            if message_id != self.current_message_id:
                return

            source = cast(DicomNode, form.cleaned_data["source"])
            assert source.node_type == DicomNode.NodeType.SERVER
            operator = DicomOperator(source.dicomserver)

            self.query_operators.append(operator)

        try:
            limit = settings.SELECTIVE_TRANSFER_RESULT_LIMIT

            studies = self.query_studies(operator, form, limit)

            received_studies: list[ResultDataset] = []
            for study in studies:
                received_studies.append(study)
                max_results_reached = len(received_studies) >= limit
                if message_id == self.current_message_id:
                    self.send_query_response(form, received_studies, max_results_reached)

            if not received_studies:
                if message_id == self.current_message_id:
                    self.send_query_response(form, received_studies, False)

        except ConnectionError:
            # Ignore connection aborts (most probably from ourself)
            # Maybe we should check here if we really aborted the connection
            pass

        finally:
            with lock:
                if operator in self.query_operators:
                    self.query_operators.remove(operator)

        return None

    def query_studies(
        self, operator: DicomOperator, form: SelectiveTransferJobForm, limit_results: int
    ) -> Iterator[ResultDataset]:
        data = form.cleaned_data

        studies = operator.find_studies(
            QueryDataset.create(
                PatientID=data["patient_id"],
                PatientName=data["patient_name"],
                PatientBirthDate=data["patient_birth_date"],
                AccessionNumber=data["accession_number"],
                StudyDate=data["study_date"],
                ModalitiesInStudy=data["modality"],
            ),
            limit_results=limit_results,
        )

        for study in studies:
            yield study

    @debounce()
    def send_query_response(
        self,
        form: SelectiveTransferJobForm,
        studies: list[ResultDataset],
        max_results_reached: bool,
    ) -> None:
        # Rerender form to remove potential previous error messages
        rendered_form = render_crispy_form(form)

        studies = sorted(
            studies,
            key=lambda study: datetime.combine(study.StudyDate, study.StudyTime),
            reverse=True,
        )

        rendered_query_results = render_to_string(
            "selective_transfer/_query_results.html",
            {
                "query": True,
                "query_results": studies,
                "max_results_reached": max_results_reached,
            },
        )

        async_to_sync(self.send)(rendered_form + rendered_query_results)

    async def make_transfer(self, form: SelectiveTransferJobForm) -> None:
        selected_studies: str | list[str] | None = form.data.get("selected_studies")
        if selected_studies is not None:
            if isinstance(selected_studies, str):
                selected_studies = [selected_studies]
            response = await self.build_transfer_response(form, selected_studies)
            await self.send(response)

    @database_sync_to_async
    def build_transfer_response(
        self, form: SelectiveTransferJobForm, selected_studies: list[str]
    ) -> str:
        rendered_form: str = render_crispy_form(form)

        try:
            job = self.transfer_selected_studies(self.user, form, selected_studies)
        except ValueError as err:
            rendered_error_message = render_error_message(str(err))
            return rendered_form + rendered_error_message

        rendered_created_job = render_to_string(
            "selective_transfer/_created_job.html",
            {"transfer": True, "created_job": job},
        )
        return rendered_form + rendered_created_job

    def transfer_selected_studies(
        self, user: User, form: SelectiveTransferJobForm, selected_studies: list[str]
    ) -> SelectiveTransferJob:
        if not selected_studies:
            raise ValueError("At least one study to transfer must be selected.")
        if len(selected_studies) > 10 and not user.is_staff:
            raise ValueError("Maximum 10 studies per selective transfer are allowed.")

        form.instance.owner = user
        job = form.save()

        pseudonym = form.cleaned_data["pseudonym"]
        for selected_study in selected_studies:
            study_data = selected_study.split("\\")
            patient_id = study_data[0]
            study_uid = study_data[1]
            SelectiveTransferTask.objects.create(
                job=job,
                source=form.cleaned_data["source"],
                destination=form.cleaned_data["destination"],
                patient_id=patient_id,
                study_uid=study_uid,
                pseudonym=pseudonym,
            )

        if user.is_staff or settings.START_SELECTIVE_TRANSFER_UNVERIFIED:
            job.status = SelectiveTransferJob.Status.PENDING
            job.save()

            job.queue_pending_tasks()

        return job
    
    async def _direct_download(self, form: SelectiveTransferJobForm) -> None:
        selected_studies: str | list[str] | None = form.data.get("selected_studies")
        if selected_studies is not None:
            if isinstance(selected_studies, str):
                selected_studies = [selected_studies]

            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self.pool, self._prepare_and_send_download_response, form, selected_studies
                )
            except HTTPError as e:
                status_code = e.response.status_code
                additional_error_text = ""
                if status_code == 400:  # Bad request
                    additional_error_text = "The source did not understand the ADIT request."
                elif status_code == 401 or status_code == 403:  # Unauthorized or forbidden
                    additional_error_text = "ADIT is not authorized to access the source."
                elif status_code == 500:  # Internal server error
                    additional_error_text = "The source could not process the ADIT request."
                elif status_code == 502:  # Bad gateway
                    additional_error_text = "The source is not available."
                form_error_response = await self._build_form_error_response(
                    form, f"Something went wrong at your requested source. {additional_error_text}"
                )
                await self.send(form_error_response)
            except (DicomError, RetriableDicomError):
                form_error_response = await self._build_form_error_response(
                    form,
                    "Something went wrong at your requested source. "
                    "A Dicom Error has occured at the source.",
                )
                await self.send(form_error_response)
            except Exception:
                form_error_response = await self._build_form_error_response(
                    form, "Something went wrong."
                )
                await self.send(form_error_response)

    def _prepare_and_send_download_response(
        self, form: SelectiveTransferJobForm, selected_studies: list[str]
    ) -> None:
        with lock:
            source = cast(DicomNode, form.cleaned_data["source"])
            assert source.node_type == DicomNode.NodeType.SERVER
            operator = DicomOperator(source.dicomserver)

            self.query_operators.append(operator)

        try:
            zip_file_name = self.prepare_download(operator, form, selected_studies)
            self.send_download_response(form, zip_file_name)

        except ConnectionError:
            # Ignore connection aborts (most probably from ourself)
            # Maybe we should check here if we really aborted the connection
            pass

        finally:
            with lock:
                if operator in self.query_operators:
                    self.query_operators.remove(operator)

        return None

    def prepare_download(
        self,
        operator: DicomOperator,
        form: SelectiveTransferJobForm,
        selected_studies: list[str],
    ) -> Path:
        download_folder = Path(settings.TEMP_DIR)
        dicom_manipulator = DicomManipulator()
        pseudonym = form.cleaned_data["pseudonym"]

        modifier = partial(
            dicom_manipulator.manipulate,
            pseudonym=pseudonym,
            trial_protocol_id=form.cleaned_data["trial_protocol_id"],
            trial_protocol_name=form.cleaned_data["trial_protocol_name"],
        )
        zipped_studies_filename = self.download_studies(
            operator,
            selected_studies,
            download_folder,
            modifier,
            pseudonym,
        )

        return zipped_studies_filename

    def download_studies(
        self,
        operator: DicomOperator,
        selected_studies: list[str],
        download_folder: Path,
        modifier: Callable,
        pseudonym: str,
    ):
        # Dynamically generate studies folder name
        prefix_studies_folder_name = "adit_selective_direct_download"
        unique_id = secrets.token_hex(4)
        current_date = datetime.now().strftime("%Y%m%d")
        studies_folder_name = f"{prefix_studies_folder_name}_{unique_id}_{current_date}_{self.user.username}"
        
        studies_folder_zip_path = download_folder / f"{studies_folder_name}.zip"
        
        study_datasets: list[Dataset] = []

        def callback(ds: Dataset | None) -> None:
            if ds is None:
                return

            modifier(ds)
            study_datasets.append(ds)
        
        with zipfile.ZipFile(studies_folder_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for selected_study in selected_studies:
                study_data = selected_study.split("\\")
                patient_id = study_data[0]
                study_uid = study_data[1]
                operator.fetch_study(
                    patient_id=patient_id,
                    study_uid=study_uid,
                    callback=callback,
                )                
                # Zip the study
                for ds in study_datasets:
                    # Determine path inside zip
                    if pseudonym:
                        patient_folder = sanitize_filename(pseudonym)
                    else:
                        patient_folder = sanitize_filename(patient_id)
                    study_date = datetime.strptime(ds.StudyDate, "%Y%m%d")
                    study_time = datetime.strptime(ds.StudyTime, "%H%M%S")
                    #modalities = ds.ModalitiesInStudy
                    #modalities = ",".join(modalities)
                    prefix = f"{study_date.strftime('%Y%m%d')}-{study_time.strftime('%H%M%S')}"
                    study_folder = f"{patient_folder}/{prefix}" #-{modalities}

                    if settings.CREATE_SERIES_SUB_FOLDERS:
                        series_folder_name = sanitize_filename(f"{ds.SeriesNumber}-{ds.SeriesDescription}")
                        internal_path = f"{study_folder}/{series_folder_name}"
                    else:
                        internal_path = study_folder

                    file_name = sanitize_filename(f"{ds.SOPInstanceUID}.dcm")
                    
                    zip_internal_path = f"{internal_path}/{file_name}"

                    stream = BytesIO()
                    write_dataset(ds, stream)
                    stream_bytes = stream.getvalue()

                    zip_file.writestr(zip_internal_path, stream_bytes)
                
                study_datasets.clear()

        return studies_folder_zip_path.name
        


    @debounce()
    def send_download_response(
        self,
        form: SelectiveTransferJobForm,
        zip_file_name,
    ) -> None:
        # Rerender form to remove potential previous error messages
        rendered_form = render_crispy_form(form)

        signer = TimestampSigner()
        token = signer.sign(zip_file_name)

        rendered_download_url = render_to_string(
            "selective_transfer/_download_url.html",
            {"download": True, "token": token, "file_name": zip_file_name},
        )

        async_to_sync(self.send)(rendered_form + rendered_download_url)
