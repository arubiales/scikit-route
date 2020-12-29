import pytest
from skroute._validators._validators import (_zero_one_validator, _intenger_validator, _float_validator,
                                          _early_stopping_validator, _validate_dict_of_dicts, _validate_route_example)
                                            

xfail = pytest.mark.xfail(strict=True)

TRUE_DICT = {1:{1:0.0,
                2:45.5,
                3:43.4},
                2:{1:45.5,
                2:0.0,
                3:56.6},
                3:{1:43.4,
                2:56.6,
                3:0.0}}

FALSE_DICT = {1:{1:0.0,
                2:45.5,
                3:43.4}}

TRUE_ROUTE = list(TRUE_DICT.keys())
FALSE_ROUTE = [6,3,2]




class TestZeroOneValidator:

    def test_zero_one_validator(self):
        assert _zero_one_validator(0.7, "test"), "Arguments of _zero_one_validator are not correct"
    
    @xfail(reason="_zero_one_validator mustn't be a number bigger than one")
    def test_zero_one_validator_fail1(self):
        assert _zero_one_validator(23.4, "test")

    @xfail(reason="_zero_one_validator mustn't be a number smaller than 0")
    def test_zero_one_validator_fail2(self):
        assert _zero_one_validator(-4.2, "test")
    
    @xfail(reason="_zero_one_validator must be a float")
    def test_zero_one_validator_fail3(self):
        assert _zero_one_validator(2, "test")
    

class TestIntengerValidator:
    
    def test_integer_validator(self):
        assert _intenger_validator(4, "test"), "The parameters of integer_validator are not correct"
    
    @xfail(reason="_intenger_validator must only accept integers")
    def test_integer_validator_fail1(self):
        _intenger_validator(4.0, "test")

    @xfail(reason="_integer_validator must only accept natural integers")
    def test_integer_validator_fail2(self):
        _intenger_validator(-7, "test")
    

class TestEarlyStoppingValidator:

    def test_early_stopping_validator(self):
        assert _early_stopping_validator(10), "The parameters of _early_stopping_validator are not correct"

    @xfail(reason="_early_stopping_validator must accept only positive integers")
    def test_early_stopping_validator_fail1(self):
        print(_early_stopping_validator(-10))

    @xfail(reason="_early_stopping_validator must accept only integers")
    def test_early_stopping_validator_fail2(self):
        _early_stopping_validator(5.4)


class TestFloatValidator:

    def test_float_validator(self):
        assert _float_validator(4.5, "test"), "The parameters of _float_validator are not correct"

    @xfail(reason="_early_stopping_validator must accept only positive floats")
    def test_float_validator_fail(self):
        print(_early_stopping_validator(-10))



class TestValidateDictOfDicts:

    def test_validate_dict_of_dicts(self, capsys):
        _validate_dict_of_dicts(TRUE_DICT, "test")
        captured = capsys.readouterr()
        assert captured.err == ""

    @xfail(reason="_validate_dict_of_dicts must return a error if the dicts of dicts not represent a simmetric matrix")
    def test_validate_dict_of_dicts_fail(self):
        _validate_dict_of_dicts(FALSE_DICT, "test")



class TestValidateRouteExample:

    def test_validate_route_example(self, capsys):
        _validate_route_example(TRUE_ROUTE, TRUE_DICT, TRUE_DICT)
        captured = capsys.readouterr()
        assert captured.err == "", "_validate_route function parameters are wrong"

    @xfail()
    def test_validate_route_example1(self):
        _validate_route_example(FALSE_ROUTE, TRUE_DICT, TRUE_DICT)
