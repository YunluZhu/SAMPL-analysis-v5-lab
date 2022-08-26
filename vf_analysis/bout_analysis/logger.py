import logging

def log_vf_ana(log_name):
    logger = logging.getLogger(f'{log_name}')
    logger.setLevel(logging.INFO)
    if not len(logger.handlers):
        fh = logging.FileHandler(f'{log_name}.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger