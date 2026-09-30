import pytest
from pydantic import BaseModel

from app.llm.structured import StructuredOutputError, extract_json, parse_model


class Person(BaseModel):
    name: str
    age: int


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_wrapped_in_prose():
    assert extract_json('这是结果：{"a": 1} 完毕') == {"a": 1}


def test_trailing_comma():
    assert extract_json('{"a": 1,}') == {"a": 1}


def test_braces_inside_string_are_ignored():
    assert extract_json('{"text": "a } b"}') == {"text": "a } b"}


def test_array_root():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_parse_model_ok():
    assert parse_model('{"name": "x", "age": 3}', Person).age == 3


def test_parse_model_invalid_raises():
    with pytest.raises(StructuredOutputError):
        parse_model('{"name": "x"}', Person)


def test_unparseable_raises():
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")
