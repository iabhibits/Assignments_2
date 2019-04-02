# IISc NLU course Assignment 2

# Seq2Seq Model with Attention for Machine Translation.

**Task:**

    Implement sequence to sequence models using LSTMs for machine translation from English to German and English to Hindi.

    Implement and compare the following kinds of encoder-decoder attention mechanisms:

    Additive attention [1]
    Multiplicative attention (without scaling) [2]
    Scaled dot-product attention [3]
    Key-value attention [4]

    Finally, implement and evaluate the effect of adding self-attention in the encoder and decoder [3].

    Use appropriate metrics to evaluate the performance of your models on the test set (for example, BLEU). Report your   observations along with your explanations for the findings.
    
    Assignment 2 : https://sites.google.com/site/2019e1246/schedule/assignment2

**Requirements:**

    nltk
    docopt
    tqdm==4.29.1
    pytorch
    numpy
    
    To install these requirements use: pip install -r gpu_requirements.txt
    

**Usage:**
    
    run.en-de.sh vocab                      Generate vocabulary and save to intermediate file.
    run.en-de.sh train                      Start training model for En-De translation.
    run.en-de.sh test                       Test trained model for En-De translation.

    run.en-hi.sh vocab                      Generate vocabulary and save to intermediate file.
    run.en-hi.sh train                      Start training model for En-Hi translation.
    run.en-hi.sh test                       Test trained model for En-Hi translation.

    use train-local / test-local for non-gpu machine.

**Fine-Tuned Usage:**

    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

**Options:**

    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 100]
    --max-epoch=<int>                       max epoch [default: 10]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --att_type=<str>                        type of attention used additive/multiplicative/key_value/scaled_dot_product [default: scaled_dot_product]
    --self_attention=<str>                  whether to use self_attention in encoder-decoder [default: False]

**Trained Model:**
    
    Final Model : Seq2Seq model with self attention.
    Trained model can be downloaded from : https://drive.google.com/open?id=1ovNsEHrabPa-Rk6f07Hkc3JPv84TLmkZ
    
**References:**

    The starter kit for project is taken from Stanford CS224N (http://web.stanford.edu/class/cs224n/)

