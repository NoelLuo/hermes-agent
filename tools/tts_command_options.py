"""Opt-in, allowlisted per-call options for operator-configured command TTS."""
from __future__ import annotations


def apply_options(config, provider, command, voice_id=None, language=None):
    """Copy a command configuration; callers can select values, never commands or paths."""
    updated = dict(command)
    for argument, key, catalog_key in ((voice_id, 'voice', 'voices'), (language, 'language', 'languages')):
        if argument is None or argument == '':
            continue
        choices = command.get(catalog_key) or {}
        if not isinstance(argument, str) or argument not in choices:
            raise ValueError(f'Unsupported TTS {key}; choose a configured {catalog_key} value')
        updated[key] = choices[argument] if isinstance(choices, dict) else argument
    providers = {**(config.get('providers') or {}), provider: updated}
    return {**config, 'providers': providers}


def schema_options(command):
    properties = {}
    for field, key in (('voice_id', 'voices'), ('language', 'languages')):
        choices = command.get(key)
        if isinstance(choices, (list, dict)) and choices:
            properties[field] = {
                'type': 'string', 'enum': list(choices),
                'description': 'Select a configured voice for this call.' if field == 'voice_id' else
                    'Output text language, not reference audio language. Omit for the voice default.'}
    return properties
