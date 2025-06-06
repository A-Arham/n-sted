# U-Net EEG Architecture for predicting a trial 

**Input shape:** `[1, 129, 374]` (Batch size = 1 for illustration)

| Layer Name        | Operation                               | Input Shape           | Output Shape          | Comments                            |
|------------------|------------------------------------------|------------------------|------------------------|--------------------------------------|
| **Encoder**       |                                          |                        |                        |                                      |
| `enc1`            | 2×Conv1d + BN + ReLU                     | `[1, 129, 374]`        | `[1, 64, 374]`         | Same padding                         |
| `pool1`           | MaxPool1d(kernel=2, stride=2)            | `[1, 64, 374]`         | `[1, 64, 187]`         |                                      |
| `enc2`            | 2×Conv1d + BN + ReLU                     | `[1, 64, 187]`         | `[1, 128, 187]`        | Same padding                         |
| `pool2`           | MaxPool1d                                | `[1, 128, 187]`        | `[1, 128, 94]`         | Ceil mode                            |
| `enc3`            | 2×Conv1d + BN + ReLU                     | `[1, 128, 94]`         | `[1, 256, 94]`         |                                      |
| `pool3`           | MaxPool1d                                | `[1, 256, 94]`         | `[1, 256, 47]`         |                                      |
| `enc4`            | 2×Conv1d + BN + ReLU                     | `[1, 256, 47]`         | `[1, 512, 47]`         |                                      |
| **Decoder**       |                                          |                        |                        |                                      |
| `up1`             | ConvTranspose1d(512→256, stride=2)       | `[1, 512, 47]`         | `[1, 256, 94]`         | Doubles temporal dim                 |
| `dec1`            | 2×Conv1d + BN + ReLU                     | `[1, 512, 94]`         | `[1, 256, 94]`         | Concatenated with `enc3` output      |
| `up2`             | ConvTranspose1d(256→128, stride=2)       | `[1, 256, 94]`         | `[1, 128, 188]`        | Ceil of 94×2                         |
| `dec2`            | 2×Conv1d + BN + ReLU                     | `[1, 256, 188]`        | `[1, 128, 188]`        | Concatenated with `enc2` output      |
| `up3`             | ConvTranspose1d(128→64, stride=2)        | `[1, 128, 188]`        | `[1, 64, 376]`         |                                      |
| `dec3`            | 2×Conv1d + BN + ReLU                     | `[1, 128, 376]`        | `[1, 64, 376]`         | Concatenated with `enc1` output      |
| `final_conv`      | Conv1d(64→1, kernel=1)                   | `[1, 64, 376]`         | `[1, 1, 376]`          | Output logits                        |
| `crop/pad`        | Align output to match input length       | `[1, 1, 376]`          | `[1, 1, 374]`          | Final alignment                      |
| `sigmoid`         | Element-wise sigmoid                     | `[1, 1, 374]`          | `[1, 1, 374]`          | Probability mask                     |



# EEG1DCNN Architecture
This CNN classifies the predicted trial (By the previous model) as MDD or not. 

**Input Shape:** `[1, 1, 384]`

| Layer Name | Type                     | Input Shape       | Output Shape      | Details                              |
|------------|--------------------------|--------------------|--------------------|--------------------------------------|
| `conv1`    | Conv1d                   | `[1, 1, 384]`      | `[1, 16, 384]`     | kernel=7, stride=1, padding=3        |
| `bn1`      | BatchNorm1d              | `[1, 16, 384]`     | `[1, 16, 384]`     |                                      |
| `pool1`    | MaxPool1d                | `[1, 16, 384]`     | `[1, 16, 192]`     | kernel=2                             |
| `drop1`    | Dropout                  | `[1, 16, 192]`     | `[1, 16, 192]`     | p = 0.3                              |
| `conv2`    | Conv1d                   | `[1, 16, 192]`     | `[1, 32, 192]`     | kernel=5, stride=1, padding=2        |
| `bn2`      | BatchNorm1d              | `[1, 32, 192]`     | `[1, 32, 192]`     |                                      |
| `pool2`    | MaxPool1d                | `[1, 32, 192]`     | `[1, 32, 96]`      | kernel=2                             |
| `drop2`    | Dropout                  | `[1, 32, 96]`      | `[1, 32, 96]`      | p = 0.3                              |
| `conv3`    | Conv1d                   | `[1, 32, 96]`      | `[1, 64, 96]`      | kernel=3, stride=1, padding=1        |
| `bn3`      | BatchNorm1d              | `[1, 64, 96]`      | `[1, 64, 96]`      |                                      |
| `pool3`    | MaxPool1d                | `[1, 64, 96]`      | `[1, 64, 48]`      | kernel=2                             |
| `drop3`    | Dropout                  | `[1, 64, 48]`      | `[1, 64, 48]`      | p = 0.3                              |
| `flatten`  | View                     | `[1, 64, 48]`      | `[1, 3072]`        | 64 × 48 = 3072                       |
| `fc1`      | Linear                   | `[1, 3072]`        | `[1, 128]`         | Fully connected                      |
| `drop_fc`  | Dropout                  | `[1, 128]`         | `[1, 128]`         | p = 0.5                              |
| `fc2`      | Linear                   | `[1, 128]`         | `[1, 2]`           | Final classification layer           |

**Output:** `[1, 2]` (class scores)



