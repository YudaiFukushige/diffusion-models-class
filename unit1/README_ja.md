# Unit 1: An Introduction to Diffusion Models

Welcome to Unit 1 of the Hugging Face Diffusion Models Course! In this unit, you will learn the basics of how diffusion 
models work and how to create your own using the 🤗 Diffusers library.

## Start this Unit :rocket:

Here are the steps for this unit:

- Make sure you've [signed up for this course](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) so that you can be notified when new material is released
- Read through the introductory material below as well as any of the additional resources that sound interesting
- Check out the _**Introduction to Diffusers**_  notebook below to put theory into practice with the 🤗 Diffusers library
- Train and share your own diffusion model using the notebook or the linked training script
- (Optional) Dive deeper with the _**Diffusion Models from Scratch**_ notebook if you're interested in seeing a minimal from-scratch implementation and exploring the different design decisions involved
- (Optional) Check out [this video](https://www.youtube.com/watch?v=09o5cv6u76c) for an informal run-through the material for this unit. 


:loudspeaker: Don't forget to join the [Discord](https://huggingface.co/join/discord), where you can discuss the material and share what you've made in the `#diffusion-models-class` channel.
 
## What Are Diffusion Models?

Diffusion models are a relatively recent addition to a group of algorithms known as 'generative models'. The goal of generative modeling is to learn to **generate** data, such as images or audio, given a number of training examples. A good generative model will create a **diverse** set of outputs that resemble the training data without being exact copies. How do diffusion models achieve this? Let's focus on the image generation case for illustrative purposes.

> 拡散モデルは，「生成モデル」と呼ばれるアルゴリズム群に比較的最近追加されたアルゴリズムである．生成モデリングの目標は，多量の訓練データを与えることで，画像や音声といったデータを生成することである．良い生成モデルは，訓練データに似ているが正確なコピーではない，多様な出力を生成することが可能である．拡散モデルはどのようにして上記の目標を達成するのだろうか？説明のために，画像生成の例に注目しよう．

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>

The secret to diffusion models' success is the iterative nature of the diffusion process. Generation begins with random noise, but this is gradually refined over a number of steps until an output image emerges. At each step, the model estimates how we could go from the current input to a completely denoised version. However, since we only make a small change at every step, any errors in this estimate at the early stages (where predicting the final output is extremely difficult) can be corrected in later updates. 

> 拡散モデル成功の秘訣は，拡散過程の性質の反復にある．生成はランダムノイズからスタートするが，出力画像が生成されるまで，何ステップもかけて徐々に精錬されてゆく．拡散モデルは各ステップで「どうすれば現状の入力からノイズ除去された画像へ移行できるか」を推定する．しかしながら，各ステップでは小さな変化しか加えられないため，最終出力を予測することが非常に困難な初期段階における推定誤差は，それより後のステップで修正できる．

Training the model is relatively straightforward compared to some other types of generative model. We repeatedly
1) Load in some images from the training data
2) Add noise, in different amounts. Remember, we want the model to do a good job estimating how to 'fix' (denoise) both extremely noisy images and images that are close to perfect.
3) Feed the noisy versions of the inputs into the model
4) Evaluate how well the model does at denoising these inputs
5) Use this information to update the model weights

> 拡散モデルの学習は，その他の生成モデルと比較すると，単純である．以下の手順が繰り返される．  
> 1. 学習データからいくつか画像を読み込む．
> 2. 様々な量のノイズを加える．このモデルには，ノイズまみれの画像とノイズが少なく完璧に近い画像の両方のノイズ除去をうまく行う方法を推定することを期待しています．
> 3. ノイズを加えた入力をモデルに与えます．
> 4. これらの入力に対して，モデルがどれだけノイズ除去に成功しているかを評価する．
> 5. この情報（4.の評価）を使って，モデルの重みをアップデートする．

To generate new images with a trained model, we begin with a completely random input and repeatedly feed it through the model, updating it each time by a small amount based on the model prediction. As we'll see, there are a number of sampling methods that try to streamline this process so that we can generate good images with as few steps as possible.

> 学習済みモデルを使って新しい画像を生成する際は，完全にランダムな入力から始めて，それを繰り返しモデルに入力し，モデルの予測に基づいて毎ステップ徐々に更新していく．このあと見るように，できるだけ少ないステップ数で良い出力を得るために，このプロセスを効率化する手法が数多く存在する．

We will show each of these steps in detail in the hands-on notebooks here in unit 1. In unit 2, we will look at how this process can be modified to add additional control over the model outputs through extra conditioning (such as a class label) or with techniques such as guidance. And units 3 and 4 will explore an extremely powerful diffusion model called Stable Diffusion, which can generate images given text descriptions.  

> この Unit 1 では，上記のステップを詳細にハンズオン形式のnotebookを使って見てゆく．Unit 2 では，このプロセスをどのように変更し，条件付け（クラスラベルなど）やガイダンスのような技術によって，モデルの出力をさらに制御することができるかを見ていくことになります．また，Unit 3 と Unit 4 では，テキストの説明から画像を生成することができる　Stable Diffusion　と呼ばれる非常に強力な拡散モデルについて調べます．


## Hands-On Notebooks

At this point, you know enough to get started with the accompanying notebooks! The two notebooks here come at the same idea in different ways. 
 
| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to Diffusers                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              |
| Diffusion Models from Scratch                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              |

In _**Introduction to Diffusers**_, we show the different steps described above using building blocks from the diffusers library. You'll quickly see how to create, train and sample your own diffusion models on whatever data you choose. By the end of the notebook, you'll be able to read and modify the example training script to train diffusion models and share them with the world! This notebook also introduces the main exercise associated with this unit, where we will collectively attempt to figure out good 'training recipes' for diffusion models at different scales - see the next section for more info.

In _**Diffusion Models from Scratch**_, we show those same steps (adding noise to data, creating a model, training and sampling) but implemented from scratch in PyTorch as simply as possible. Then we compare this 'toy example' with the diffusers version, noting how the two differ and where improvements have been made. The goal here is to gain familiarity with the different components and the design decisions that go into them so that when you look at a new implementation you can quickly identify the key ideas.

## Project Time

Now that you've got the basics down, have a go at training one or more diffusion models! Some suggestions are included at the end of the _**Introduction to Diffusers**_ notebook. Make sure to share your results, training recipes and findings with the community so that we can collectively figure out the best ways to train these models.

## Some Additional Resources
 
[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) is a very in-depth walk-through of the code and theory behind DDPMs with 
 maths and code showing all the different components. It also links to a number of papers for further reading.
 
Hugging Face documentation on [Unconditional Image-Generation](https://huggingface.co/docs/diffusers/training/unconditional_training) for some examples of how to train diffusion models using the official training example script, including code showing how to create your own dataset. 

AI Coffee Break video on Diffusion Models: https://www.youtube.com/watch?v=344w5h24-h8

Yannic Kilcher Video on DDPMs: https://www.youtube.com/watch?v=W-O7AZNzbzQ

Found more great resources? Let us know and we'll add them to this list.
