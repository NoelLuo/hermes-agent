"""Retained-image lookup scales with transcript size and preserves path spellings."""
import json
import sqlite3

import pytest

from gateway.session_ingress_media import _history_mentions


def test_many_candidates_do_not_multiply_the_transcript_query(tmp_path):
    references = [
        {'sha256': f'{index:064x}', 'path': str(tmp_path / f'{index:064x}' / 'image.png')}
        for index in range(1010)
    ]
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)')
        conn.executemany('INSERT INTO messages(content) VALUES(?)',
                         [('unrelated ordinary conversation',)] * 200)
        last = references[-1]
        conn.execute('INSERT INTO messages(content) VALUES(?)',
                     (last['sha256'] + '/image.png',))
        # A bounded SQLite VM budget rejects a candidates x messages nested scan, independent
        # of host speed. The expected answer is still the last candidate in the last row.
        callbacks = 0
        def budget():
            nonlocal callbacks
            callbacks += 1
            return callbacks > 200
        conn.set_progress_handler(budget, 100)
        try:
            assert _history_mentions(conn, references) == {last['path']}
        finally:
            conn.set_progress_handler(None, 0)


@pytest.mark.parametrize('separator', ['/', '\\'])
@pytest.mark.parametrize('escaped', [False, True])
def test_history_path_spellings_and_snapshot_watermark(tmp_path, separator, escaped):
    digest = 'ab' * 32
    reference = {'sha256': digest, 'path': str(tmp_path / digest / '图 image.png')}
    tail = digest + separator + '图 image.png'
    content = json.dumps(tail)[1:-1] if escaped else tail
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)')
        conn.execute('INSERT INTO messages VALUES(1, ?)', ('older: ' + content,))
        conn.execute('INSERT INTO messages VALUES(2, ?)', (digest + '/other.png',))
        assert _history_mentions(conn, [reference]) == {reference['path']}
        assert _history_mentions(conn, [reference], after_id=1) == set()
        conn.execute('INSERT INTO messages VALUES(3, ?)', ('new: ' + content,))
        assert _history_mentions(conn, [reference], after_id=1) == {reference['path']}
