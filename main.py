from model import PopMusicTransformer
import os
# import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# tf.compat.v1.disable_eager_execution()


def main():
    # declare model
    model = PopMusicTransformer(checkpoint='/mnt/workspace/tangxinyu/fa_music/remi/REMI-finetune', is_training=False)

    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch_finetune_174.midi',
        prompt=None,
    )

    # generate continuation
    # model.generate(
    #     n_target_bar=16,
    #     temperature=1.2,
    #     topk=5,
    #     output_path='./result/continuation_1.midi',
    #     prompt='./data/evaluation/000.midi',
    # )

    # close model
    model.close()


if __name__ == '__main__':
    main()
