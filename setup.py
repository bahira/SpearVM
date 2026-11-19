"""Include native C sources in installed source-only Linux/macOS wheels.

The upstream C source remains src/spur_kernels.c; build copies it into the
package's build output so spur_math.__init__ can compile on first import
when a prebuilt native shared object is not bundled.
"""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class NativeSourceBuildPy(_build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent / 'src' / 'spur_kernels.c'
        if not source.is_file():
            raise RuntimeError('Missing native C source src/spur_kernels.c in sdist')
        target = Path(self.build_lib) / 'spur_math' / 'src' / 'spur_kernels.c'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        outputs.append(str(Path(self.build_lib) / 'spur_math' / 'src' /
                           'spur_kernels.c'))
        return outputs


setup(cmdclass={'build_py': NativeSourceBuildPy})
