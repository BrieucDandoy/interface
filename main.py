from input_manager import input_manager
import logging
root_logger = logging.getLogger()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = "main.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def main():
    print('Initialisazing ...')
    manager = input_manager("D:/Documents/model/mistral248M")
    print('Enter your input :\n')
    while manager:
        user_input = input()
        message = manager.get(user_input)
        if message is not None:
            print(message,'\n')


if __name__ == '__main__':
    main()