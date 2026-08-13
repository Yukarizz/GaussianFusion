from .models import register, make
from . import edsr
from . import gaussian
from . import mlp
from . import cnn
from . import unet
from . import rdn
from . import swinir
from . import spynet
from . import searaft
from . import fusion
from . import temporal_attention
from . import gaussian_fusion

# HAT depends on the (outdated) basicsr package which is incompatible with newer
# torchvision. It is only needed if a config explicitly uses `name: hat`, so we
# import it lazily and skip it when basicsr cannot be imported.
try:
    from . import hat
except Exception as _e:  # pragma: no cover
    print(f'[models] WARNING: HAT not imported (basicsr unavailable/incompatible): {_e}')
