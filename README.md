Embedding Watermarks into Deep Neural Networks
====
This code is the pytorch implementation of "Embedding Watermarks into Deep Neural Networks" [1]. It embeds a digital watermark into deep neural networks in training the host network. This embedding is achieved by a parameter regularizer.

## Usage
Embed a watermark in training a host network:

```sh
# train the host network while embedding a watermark
python train_watermark.py config/train_random_min.json

# extract the embedded watermark
python val_watermark.py result/wrn_WTYPE_random_DIM256_SCALE0.01_N1K4B64EPOCH3_TBLK1.weight result/wrn_WTYPE_random_DIM256_SCALE0.01_N1K4B64EPOCH3_TBLK1_layer7_w.npy result/random
```

Train the host network *without* embedding:

```sh
# train the host network without embedding
python train_wrn.py config/train_non_min.json 

# extract the embedded watermark (meaningless because no watermark was embedded)
python val_watermark.py result/wrn_WTYPE_random_DIM256_SCALE0.01_N1K4B64EPOCH3_TBLK0.weight result/wrn_WTYPE_random_DIM256_SCALE0.01_N1K4B64EPOCH3_TBLK1_layer7_w.npy result/non

# visualize the embedded watermark
python utility/draw_histogram_signature.py config/draw_histogram_non.json hist_signature_non.png
```

## References
[1] Y. Uchida, Y. Nagai, S. Sakazawa, and S. Satoh, "Embedding Watermarks into Deep Neural Networks," ICMR, 2017.