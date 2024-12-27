import sys
import traceback
import logging
from logstash.formatter import LogstashFormatterVersion1

message_type="diffbot-llm"

class LogStashCustomFormatter(LogstashFormatterVersion1):
    def format(self, record):
        record.path = None # clear the path information
        return super().format(record)

class ConsoleCustomFormatter(LogStashCustomFormatter):
    def format(self, record):
        formatted_record = super().format(record).decode('utf-8')
        if record.exc_info:
            exception_string = ''.join(traceback.format_exception(*record.exc_info))
            print(exception_string, file=sys.stderr)

        return formatted_record   

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ConsoleCustomFormatter(message_type=message_type))

handler = console_handler
loggers = {}
def get_logstash_logger(logger_name:str = None):
    if logger_name is None or logger_name == '':
        logger_name = 'diffbot-llm'
        
    if logger_name not in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        loggers[logger_name] = logger

    return loggers[logger_name]