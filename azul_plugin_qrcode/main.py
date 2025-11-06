"""Decode qrcodes from documents and images."""

import io
import logging
import re
import zipfile
from urllib.parse import urlparse

import fitz  # PyMuPDF
from azul_bedrock import models_network as azm
from azul_runner import (
    BinaryPlugin,
    Feature,
    FeatureType,
    Job,
    State,
    add_settings,
    cmdline_run,
)
from PIL import Image, UnidentifiedImageError
from pyzbar.pyzbar import decode

# This plugin won't get qr codes that are not embedded within an excel sheet, e.g images copy pasted
# and not in the /media directory will not be found. The could be located if the workbook was loaded and
# sheets scanned, however this seems like overkill for this plugin.

logger = logging.getLogger(__name__)


class AzulPluginQrcode(BinaryPlugin):
    """Decode qrcodes from documents and images."""

    VERSION = "2025.09.16"

    # List of filetypes
    office_filetypes = [
        "document/office/excel",
        "document/office/word",
        "document/office/powerpoint",
        "document/office/unknown",
    ]
    pdf_filetypes = ["document/pdf"]
    email_filetypes = ["document/email", "document/office/email"]
    img_filetype = "image/"

    SETTINGS = add_settings(
        filter_data_types={"content": office_filetypes + pdf_filetypes + email_filetypes + [img_filetype]},
    )

    FEATURES = [
        Feature(
            "qr_code_data_raw", desc="The raw data from the qr code, including formatting", type=FeatureType.String
        ),
        Feature("qr_code_type", desc="The type, if provided", type=FeatureType.String),
        Feature("qr_code_rect", desc="The rectangle bounds of the qr code", type=FeatureType.String),
        Feature("qr_code_polygon", desc="The polygon bounds of the qr code", type=FeatureType.String),
        Feature("qr_code_quality", desc="The quality of the qr code", type=FeatureType.String),
        Feature("qr_code_orientation", desc="The orientation of the qr code", type=FeatureType.String),
        Feature("qr_code_uri", desc="URI's found within the qr code", type=FeatureType.Uri),
        Feature("qr_code_email", desc="Email addresses found within the qr code", type=FeatureType.String),
    ]

    images_processed = 0

    def process_image(self, img):
        """Processes images, attempting to extract qr codes and saves relevant features."""
        # Images processed represets attempts to process an image rather than successful completions
        self.images_processed += 1
        data = decode(img)
        if len(data) > 0:
            for raw in data:

                # Try to decode to utf-8
                try:
                    decoded_data = raw.data.decode("utf-8")
                except Exception as e:
                    logger.error(f"azul_qr: failed to decode to utf-8: {e}")
                    self._event_main.add_child_with_data(
                        relationship={"action": "extracted_qr_code"}, data=raw.data, label=azm.DataLabel.TEXT
                    )

                # grab anything that looks like a url
                candidates = re.findall(r"\S+://\S+", decoded_data)
                for c in candidates:
                    if urlparse(c):
                        self.add_feature_values("qr_code_uri", c)

                # grab emails
                emails = re.findall(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", raw.data.decode("utf-8"))
                for e in emails:
                    self.add_feature_values("qr_code_email", e)

                if len(raw.data) > self.cfg.max_value_length:
                    self._event_main.add_text(str(raw.data))
                    self.add_feature_values(
                        "qr_code_data_raw", decoded_data[0 : self.cfg.max_value_length - 3] + "..."
                    )
                else:
                    self.add_feature_values("qr_code_data_raw", decoded_data)

                if raw.type:
                    self.add_feature_values("qr_code_type", str(raw.type))
                if raw.rect:
                    self.add_feature_values("qr_code_rect", str(raw.rect))
                if raw.polygon:
                    self.add_feature_values("qr_code_polygon", str(raw.polygon))
                if raw.quality:
                    self.add_feature_values("qr_code_quality", str(raw.quality))
                if raw.orientation:
                    self.add_feature_values("qr_code_orientation", str(raw.orientation))

    def extract_images_from_office(self, path):
        """Given a office file extracts images that are embedded in the media directories."""
        # office files are really just zip archives
        try:
            with zipfile.ZipFile(path, "r") as zip:

                # Images are stored in 'word/media' inside the archive
                paths = ("word/media/", "ppt/media/", "xl/media/", "media/")
                image_files = [f for f in zip.namelist() if (f.startswith(paths))]
                if not image_files:
                    return

                for image_file in image_files:
                    if self.images_processed >= 100:
                        return State(
                            State.Label.COMPLETED_WITH_ERRORS,
                            "Office has more than 100 images, only processed first 100",
                        )

                    # the loop captures 'word/media' etc, so remove the paths with no image
                    if image_file in paths:
                        continue

                    image_data = zip.read(image_file)
                    out_img = Image.open(io.BytesIO(image_data)).convert("RGB")
                    self.process_image(out_img)

        except zipfile.BadZipFile as e:
            logger.error(f"auzl_qr: error processing office, bad zip file: {e}")
        except UnidentifiedImageError:
            logger.error("auzl_qr: error processing office, can't open image:")
        except OSError as e:
            if "cannot load this image" in str(e):
                logger.error("azul_qr: OSError, couldn't load image")
            else:
                raise OSError(f"auzl_qr: uncaught OSError converting image: {e}")

    # look for more efficient why to process.
    def extract_images_from_pdf(self, path):
        """Given a pdf file extracts images."""
        try:
            with fitz.open(path) as pdf:

                for xref in range(1, pdf.xref_length()):
                    try:
                        if self.images_processed >= 100:
                            return State(
                                State.Label.COMPLETED_WITH_ERRORS,
                                "PDF has more than 100 images, only processed first 100",
                            )
                        info = pdf.xref_object(xref, compressed=False)

                        if "/Image" in info:  # crude filter for image objects, could be improved
                            base_image = pdf.extract_image(xref)["image"]
                            out_img = Image.open(io.BytesIO(base_image)).convert("RGB")
                            self.process_image(out_img)

                    except ValueError as e:
                        logger.error(f"azul_qr: Conversion error processing pdf image: {e}")

        except RuntimeError as e:
            msg = str(e)
            if "cannot open broken document" in msg:
                logger.error("The file is corrupted or not a valid PDF.")
            elif "no such file" in msg:
                logger.error("The file does not exist.")
            elif "cannot open encrypted document" in msg:
                logger.error("The file is password-protected.")
            else:
                raise RuntimeError(f"auzl_qr: uncaught error processing pdf: {e}")
        except UnidentifiedImageError:
            logger.error("auzl_qr: error processing office, can't open image")
        except OSError as e:
            if "image file is truncated" in str(e):
                logger.error("azul_qr: OSError, the file is corrupted or not a valid PDF.")
            elif "cannot load this image" in str(e):
                logger.error("azul_qr: OSError, couldn't load image")
            else:
                raise OSError(f"auzl_qr: uncaught OSError converting image: {e}")

    def execute(self, job: Job):
        """Run the plugin."""
        self.images_processed = 0
        self._event_main

        path = job.get_data().get_filepath()
        file_format = job.event.entity.datastreams[0].file_format

        # based on the filetype run image extractors
        if file_format in self.office_filetypes:
            return self.extract_images_from_office(path)
        elif file_format in self.pdf_filetypes:
            return self.extract_images_from_pdf(path)
        elif file_format.startswith(self.img_filetype):
            # convert to an image - the try catch will capture issues with some image types that cannot be convereted.
            try:
                self.process_image(Image.open(io.BytesIO(job.get_data().read())).convert("RGB"))
            except UnidentifiedImageError:
                logger.error("auzl_qr: error processing office, can't open image:")
            except OSError as e:
                if "image file is truncated" in str(e):
                    logger.error("azul_qr: OSError, the file is corrupted or not a valid PDF.")
                elif "cannot load this image" in str(e):
                    logger.error("azul_qr: OSError, couldn't load image")
                else:
                    raise OSError(f"auzl_qr: uncaught OSError converting image: {e}")
        else:
            # not able to identify the type, try various types

            # The return here should not cause issues, if the file is not processable as word
            # it should error and continue, trying other file types. The return is used to pass
            # back the error if there are more than 100 images to process.
            try:
                return self.extract_images_from_office(path)
            except Exception as e:
                logger.error(f"auzl_qr: could not detect file type, word failed: {e}")
            if self.images_processed > 0:
                return

            try:
                self.process_image(Image.open(io.BytesIO(job.get_data().read())).convert("RGB"))
            except UnidentifiedImageError:
                logger.error("auzl_qr: could not detect file type: image conversion failed.")
            if self.images_processed > 0:
                return

            try:
                return self.extract_images_from_pdf(path)
            except Exception as e:
                logger.error(f"auzl_qr: could not detect file type, pdf failed: {e}")

            return State(State.Label.OPT_OUT, "Unable to process file type")


def main():
    """Plugin command-line entrypoint."""
    cmdline_run(plugin=AzulPluginQrcode)


if __name__ == "__main__":
    main()
