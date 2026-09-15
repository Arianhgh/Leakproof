"""imbalanced-learn adapter — resamplers/augmenters that must run post-split."""

from __future__ import annotations

from . import FrameworkAdapter

_RESAMPLERS = {
    "SMOTE",
    "SMOTENC",
    "SMOTEN",
    "BorderlineSMOTE",
    "KMeansSMOTE",
    "SVMSMOTE",
    "ADASYN",
    "RandomOverSampler",
    "RandomUnderSampler",
    "NearMiss",
    "TomekLinks",
    "EditedNearestNeighbours",
    "ClusterCentroids",
    "SMOTETomek",
    "SMOTEENN",
}


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="imblearn",
        resamplers=_RESAMPLERS,
        pipeline_constructors={"Pipeline", "make_pipeline"},  # imblearn.pipeline
        learn_methods={"fit", "fit_resample", "fit_transform"},
        apply_methods={"resample", "transform"},
        seeded_callables=_RESAMPLERS,
    )
