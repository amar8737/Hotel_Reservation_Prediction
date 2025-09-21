import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message, errors_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message
    @staticmethod
    def get_error_message(error_message, errors_detail):
        errors_detail = errors_detail.exc_info()
        _, _, exc_tb = errors_detail
        line_number = exc_tb.tb_lineno if exc_tb else None
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None
        error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"
        return error_message
    def __str__(self):
        return self.error_message