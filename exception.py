import sys,os


def get_error_msg(error,error_detail:sys):

    _,_,exc_tb=error_detail.exc_info()

    if exc_tb is not None:
        filename=exc_tb.tb_frame.f_code.co_filename
        lineno=exc_tb.tb_lineno
    else:
        filename="unknown"
        lineon="unknown"

    return f"\nFILE  [{filename}]\nLINE  [{lineno}]\nERROR  [{str(error)}]"


class CustomException(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)

        self.custom_error_msg=get_error_msg(error_msg,error_detail)

    def __str__(self) -> str:
        return self.custom_error_msg


if __name__=="__main__":

    try:
        a=4/0
    except Exception as e:
        raise CustomException(e,sys)

    