from gettext import npgettext
from models.transformer import SwinTransformer
import numpy as np

from models.model_t import sm_transformer_pm

model = SwinTransformer('teste_128', num_classes=1000, include_top=True, pretrained=False)

inp = np.random.rand(1000, 128, 128, 3)
label = np.random.rand(1000, 1000)

model.compile(
    loss = 'binary_crossentropy',
    run_eagerly = True
)

a = model.fit(
    x = inp,
    y = label,
    batch_size = 32,
    validation_split = 0.2,
    epochs = 10
)

print(model.summary())