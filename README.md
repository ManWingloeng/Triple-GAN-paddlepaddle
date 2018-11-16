# TripleGAN-paddlepaddle
Simple paddlepaddle implementation of [Triple Generative Adversarial Nets](https://arxiv.org/pdf/1703.02291.pdf)(Triple-GAN)

If you want to see the original author's code, please refer to this [link](https://github.com/zhenxuan00/triple-gan)


## Usage
```bash
> python triple_gan.py
```


## Idea
### Network Architecture
![network](./assests/network.JPG)

### Algorithm
![algorithm](./assests/algorithm.JPG)

## Result
### Classification result
![c_result](./assests/result.JPG)

### Convergence speed on SVHN
![s_result](./assests/result2.JPG)

## My result (Cifar10, 4000 labelled image)
### Loss
![loss](./assests/loss.png)

### Classification accuracy
![accuracy](./assests/accuracy.png)

### Generated Image (Other images are in assests)
#### Automobile
![automobile](./assests/generated_image/class_1.png)

## Related works
* [CycleGAN](https://github.com/taki0112/CycleGAN-Tensorflow)
* [DiscoGAN](https://github.com/taki0112/DiscoGAN-Tensorflow)
* [TripleGAN-Tensorflow](https://github.com/taki0112/TripleGAN-Tensorflow)

## Reference
* [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

## Author
Todd
