import re
from targon.core.exceptions import ValidationError


def check_object_name(name: str):
    if not isinstance(name, str):
        raise ValidationError("Object name must be a string", field="name", value=name)
    if len(name) < 3 or len(name) > 63:
        raise ValidationError(
            "App name must be between 3 and 63 characters long",
            field="name",
            value=name,
        )
    if not re.match(r"^[a-z0-9-]+$", name):
        raise ValidationError(
            "Object name may only contain lowercase letters, numbers, and hyphens",
            field="name",
            value=name,
        )
    if name.startswith("-") or name.endswith("-") or "--" in name:
        raise ValidationError(
            "Object name cannot start/end with a hyphen or contain consecutive hyphens",
            field="name",
            value=name,
        )
