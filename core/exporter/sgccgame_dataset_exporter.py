# -*- coding: utf-8 -*-
'''
@Author: captainsama
@Date: 2023-02-27 16:20:02
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-28 18:36:11
@FilePath: /dataset_manager/core/exporter/sgccgame_dataset_exporter.py
@Description:
'''
import os
import fiftyone.core.metadata as fom
import fiftyone.utils.data as foud
import fiftyone.utils.voc as fouvoc
import fiftyone.core.labels as fol


class SGCCGameDatasetExporter(fouvoc.VOCDetectionDatasetExporter):
    """Exporter that writes VOC detection datasets to disk.

    See :ref:`this page <VOCDetectionDataset-export>` for format details.

    Args:
        export_dir (None): the directory to write the export. This has no
            effect if ``data_path`` and ``labels_path`` are absolute paths
        data_path (None): an optional parameter that enables explicit control
            over the location of the exported media. Can be any of the
            following:

            -   a folder name like ``"data"`` or ``"data/"`` specifying a
                subfolder of ``export_dir`` in which to export the media
            -   an absolute directory path in which to export the media. In
                this case, the ``export_dir`` has no effect on the location of
                the data
            -   a JSON filename like ``"data.json"`` specifying the filename of
                the manifest file in ``export_dir`` generated when
                ``export_media`` is ``"manifest"``
            -   an absolute filepath specifying the location to write the JSON
                manifest file when ``export_media`` is ``"manifest"``. In this
                case, ``export_dir`` has no effect on the location of the data

            If None, the default value of this parameter will be chosen based
            on the value of the ``export_media`` parameter
        labels_path (None): an optional parameter that enables explicit control
            over the location of the exported labels. Can be any of the
            following:

            -   a folder name like ``"labels"`` or ``"labels/"`` specifying the
                location in ``export_dir`` in which to export the labels
            -   an absolute folder path to which to export the labels. In this
                case, the ``export_dir`` has no effect on the location of the
                labels

            If None, the labels will be exported into ``export_dir`` using the
            default folder name
        export_media (None): controls how to export the raw media. The
            supported values are:

            -   ``True``: copy all media files into the output directory
            -   ``False``: don't export media
            -   ``"move"``: move all media files into the output directory
            -   ``"symlink"``: create symlinks to the media files in the output
                directory
            -   ``"manifest"``: create a ``data.json`` in the output directory
                that maps UUIDs used in the labels files to the filepaths of
                the source media, rather than exporting the actual media

            If None, the default value of this parameter will be chosen based
            on the value of the ``data_path`` parameter
        rel_dir (None): an optional relative directory to strip from each input
            filepath to generate a unique identifier for each image. When
            exporting media, this identifier is joined with ``data_path`` and
            ``labels_path`` to generate output paths for each exported image
            and labels file. This argument allows for populating nested
            subdirectories that match the shape of the input paths. The path is
            converted to an absolute path (if necessary) via
            :func:`fiftyone.core.utils.normalize_path`
        include_paths (True): whether to include the absolute paths to the
            images in the ``<path>`` elements of the exported XML
        image_format (None): the image format to use when writing in-memory
            images to disk. By default, ``fiftyone.config.default_image_ext``
            is used
        extra_attrs (True): whether to include extra object attributes in the
            exported labels. Supported values are:

            -   ``True``: export all extra attributes found
            -   ``False``: do not export extra attributes
            -   a name or list of names of specific attributes to export
    """

    def __init__(
        self,
        export_dir=None,
        data_path=None,
        labels_path=None,
        export_media=None,
        rel_dir=None,
        include_paths=True,
        image_format=None,
        extra_attrs=False,
    ):
        super().__init__(
            export_dir,
            data_path,
            labels_path,
            export_media,
            rel_dir,
            include_paths,
            image_format,
            extra_attrs,
        )
        self._export_dir = export_dir

    def export_sample(self, image_or_path, detections, metadata=None):
        out_image_path, uuid = self._media_exporter.export(image_or_path)

        if detections is None:
            return

        out_labels_path = os.path.join(
            self.labels_path, os.path.splitext(uuid)[0] + ".xml"
        )

        if metadata is None:
            metadata = fom.ImageMetadata.build_for(image_or_path)

        if self.include_paths:
            path = out_image_path
        else:
            path = None

        annotation = fouvoc.VOCAnnotation.from_labeled_image(
            metadata,
            detections,
            path=path,
            filename=uuid,
            extra_attrs=self.extra_attrs,
        )
        self._writer.write(annotation, out_labels_path)