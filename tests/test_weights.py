"""Checkpoint resolution must not substitute published weights for custom paths."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from model import weights


class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.patch = patch.object(weights, 'WEIGHTS_DIR', self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.download = Mock()
        self.module = patch.dict(sys.modules, {
            'huggingface_hub': types.SimpleNamespace(hf_hub_download=self.download),
        })
        self.module.start()
        self.addCleanup(self.module.stop)

    def test_existing_custom_file_needs_no_download(self):
        path = self.directory / 'custom.pt'
        path.write_bytes(b'local')
        self.assertEqual(weights.resolve_weights(path), path)
        self.download.assert_not_called()

    def test_missing_release_downloads_pinned_revision(self):
        path = self.directory / 'DMM-08M.pt'
        self.download.return_value = str(path)
        self.assertEqual(weights.resolve_weights(path), path)
        self.download.assert_called_once_with(
            repo_id=weights.REPO_ID, filename=path.name,
            revision=weights.REVISION, local_dir=self.directory,
        )

    def test_missing_custom_paths_never_download(self):
        for path in (self.directory / 'typo.pt', self.directory / 'custom' / 'DMM-08M.pt'):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                weights.resolve_weights(path)
        self.download.assert_not_called()

    def test_failed_download_has_recovery_instructions(self):
        self.download.side_effect = OSError('offline')
        with self.assertRaisesRegex(RuntimeError, 'HF_HUB_OFFLINE=1') as caught:
            weights.resolve_weights(self.directory / 'DMM-08M.pt')
        self.assertIsInstance(caught.exception.__cause__, OSError)


if __name__ == '__main__':
    unittest.main()
