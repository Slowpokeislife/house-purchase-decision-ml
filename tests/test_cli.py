from house_purchase.cli import main
import pytest

def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0
    assert 'tune' in capsys.readouterr().out
