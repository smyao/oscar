"""Archive #86/#170/#174: synthetic publication/cache semantics, no NPU.

Synthetic certificate metrics exercise publication, loading and reuse only;
they are not numerical/model-performance evidence. No production edit.
"""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_optr_experimental_publication import failed_experiment
from oscar_ascend.optr_artifacts import publish_artifact, validate_refinement_metadata, artifact_test_status
from oscar_ascend.optr_checkpoint import FitCheckpoints
from oscar_ascend.rotations import rotation_artifact_path
from tools.prepare_optr_rotations import prepare


def forbidden(*args, **kwargs):
    raise AssertionError('cache unexpectedly recomputed fitting/capture')


def main():
    with tempfile.TemporaryDirectory(prefix='optr_diag_cert_cache_') as directory:
        root = Path(directory)
        config, artifact = failed_experiment(root)
        certificate = validate_refinement_metadata(artifact)
        destination = rotation_artifact_path(config, artifact['model_fingerprint'], root=root)
        publish_artifact(artifact, destination)
        with patch('oscar_ascend.optr_checkpoint.fitting_implementation_digest', forbidden):
            reused = prepare(config, output=destination, testing=True,
                base_runner=forbidden, fitter=forbidden, capture_runner=forbidden)
        old = FitCheckpoints(root / 'fits', {'profile': artifact['calibration_profile'],
                             'fitting_implementation_sha256': 'old-implementation'})
        new = FitCheckpoints(root / 'fits', {'profile': artifact['calibration_profile'],
                             'fitting_implementation_sha256': 'fixed-implementation'})
        result = {'kind': 'synthetic_certificate_semantics_not_model_accuracy',
            'failed_independent_test_certificate_validated': certificate['independent_test'],
            'artifact_test_status': artifact_test_status(artifact),
            'published_artifact_loaded_and_reused': reused,
            'complete_artifact_cache_does_not_consult_fitter_implementation_digest': True,
            'partial_fit_checkpoint_does_change_identity': old.root != new.root}
        assert reused['reused'] and reused['independent_test'] == 'failed'
        assert old.root != new.root
        Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'published_artifact_loaded_and_reused'}, indent=2))


if __name__ == '__main__':
    main()
