from model import PopMusicTransformer
from glob import glob
import os
from logger_config import get_logger,  setup_logging 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    output_checkpoint_folder = 'REMI-chord-finetune' # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)
    # 设置日志
    logger = setup_logging(output_checkpoint_folder)
    logger.info("Starting the fine-tuning process")

    # 声明模型
    logger.info("Initializing PopMusicTransformer model")
    model = PopMusicTransformer(
        checkpoint='./oss_remi/REMI-tempo-chord-checkpoint',
        is_training=True)
    logger.info("Model initialized successfully")

    # prepare data
    midi_paths = glob('./oss_remi/data/train/*.midi') # you need to revise it
    logger.info(f"Found {len(midi_paths)} MIDI files")
    logger.info("Preparing training data")
    training_data = model.prepare_data(midi_paths=midi_paths)
    logger.info(f"Training data prepared, shape: {training_data.shape}")


    # check output checkpoint folder
    ####################################
    # if you use "REMI-tempo-chord-checkpoint" for the pre-trained checkpoint
    # please name your output folder as something with "chord"
    # for example: my-love-chord, cute-doggy-chord, ...
    # if use "REMI-tempo-checkpoint"
    # for example: my-love, cute-doggy, ...
    ####################################
    
    
    # finetune
    logger.info(f"Starting fine-tuning, output folder: {output_checkpoint_folder}")
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)
    logger.info("Fine-tuning completed")

    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into the your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model.close()
    logger.info("Model closed. Process completed.")

if __name__ == '__main__':
    main()
