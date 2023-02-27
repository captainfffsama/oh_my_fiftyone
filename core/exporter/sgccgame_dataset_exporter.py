# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-02-27 16:20:02
@LastEditors: captainsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-27 16:25:36
@FilePath: /dataset_manager/core/exporter/sgccgame_dataset_exporter.py
@Description:
'''
import fiftyone.utils.data as foud
import fiftyone.core.labels as fol

class SGCCGameDatasetExporter(foud.LabeledImageDatasetExporter):
    """Custom exporter for labeled image datasets.

    Args:
        export_dir (None): the directory to write the export. This may be
            optional for some exporters
        **kwargs: additional keyword arguments for your exporter
    """

    def __init__(self, export_dir=None, **kwargs):
        super().__init__(export_dir=export_dir)
        self._export_dir=export_dir

    @property
    def requires_image_metadata(self):
        """Whether this exporter requires
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
        being exported.
        """
        # Return True or False here
        True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) exported by this
        exporter.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            exporter directly exports labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the exporter can export a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the exporter can handle label dictionaries with
            value-types specified by this dictionary. Not all keys need be
            present in the exported label dicts
        -   ``None``. In this case, the exporter makes no guarantees about the
            labels that it can export
        """
        # Return the appropriate value here
        return fol.Detections

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
        the dataset.

        This method is called when the exporter's context manager interface is
        entered, :func:`DatasetExporter.__enter__`.
        """
        # Your custom setup here
        pass

    def log_collection(self, sample_collection):
        """Logs any relevant information about the
        :class:`fiftyone.core.collections.SampleCollection` whose samples will
        be exported.

        Subclasses can optionally implement this method if their export format
        can record information such as the
        :meth:`fiftyone.core.collections.SampleCollection.name` and
        :meth:`fiftyone.core.collections.SampleCollection.info` of the
        collection being exported.

        By convention, this method must be optional; i.e., if it is not called
        before the first call to :meth:`export_sample`, then the exporter must
        make do without any information about the
        :class:`fiftyone.core.collections.SampleCollection` (which may not be
        available, for example, if the samples being exported are not stored in
        a collection).

        Args:
            sample_collection: the
                :class:`fiftyone.core.collections.SampleCollection` whose
                samples will be exported
        """
        # Log any information from the sample collection here
        pass

    def export_sample(self, image_or_path, label, metadata=None):
        """Exports the given sample to the dataset.

        Args:
            image_or_path: an image or the path to the image on disk
            label: an instance of :meth:`label_cls`, or a dictionary mapping
                field names to :class:`fiftyone.core.labels.Label` instances,
                or ``None`` if the sample is unlabeled
            metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
                instance for the sample. Only required when
                :meth:`requires_image_metadata` is ``True``
        """
        # Export the provided sample
        pass

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        exported.

        This method is called when the importer's context manager interface is
        exited, :func:`DatasetExporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetExporter.__exit__`
        """
        # Your custom code here to complete the export
        pass