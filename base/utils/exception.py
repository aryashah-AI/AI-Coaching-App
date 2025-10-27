import sys

from base.utils.logger import logger


class AppServices:
    @staticmethod
    def handle_exception(exception, is_raise=False):
        exc_type, _, tb = sys.exc_info()
        f = tb.tb_frame
        line_no, filename, function_name = \
            tb.tb_lineno, f.f_code.co_filename, f.f_code.co_name

        message = exception.detail if hasattr(exception, "detail") and bool(
            exception.detail) \
            else f"Exception type: {exc_type}, " \
                 f"Exception message: {exception.__str__()}, " \
                 f"Filename: {filename}, " \
                 f"Function name: {function_name} " \
                 f"Line number: {line_no}"

        logger.error(f"Exception error message: {message}")
