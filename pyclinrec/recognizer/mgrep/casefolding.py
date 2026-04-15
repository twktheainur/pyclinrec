import struct


class CaseFolding:
    def __init__(self, source):
        self.code_arr = (
            []
        )  # Stores tuples of (source unicode, start location, end location)
        self.mapping_buf = bytearray()
        self.load_data(source)

    def load_data(self, source):
        if isinstance(source, str):
            with open(source, "r") as file:
                self._parse_file(file)
        elif hasattr(source, "read"):
            self._load_from_file(source)
        else:
            raise ValueError("Source must be a filename or a file object")

    def _parse_file(self, file):
        line_num = 0
        for line in file:
            line_num += 1
            line = line.strip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 3:
                raise ValueError(f"Line {line_num} parsing error")
            status = parts[1].strip()
            if status in ("C", "S"):
                source_unicode = int(parts[0], 16)
                mapping = [int(x, 16) for x in parts[2].split(" ")]
                self._add_mapping(source_unicode, mapping)
            elif status not in ("F", "T"):
                raise ValueError(f"Line {line_num} column 2 is not C/F/S/T")

    def _add_mapping(self, source_unicode, mapping):
        start_location = len(self.mapping_buf)
        for code in mapping:
            self.mapping_buf.extend(code.to_bytes((code.bit_length() + 7) // 8, "big"))
        end_location = len(self.mapping_buf)
        self.code_arr.append((source_unicode, start_location, end_location))

    def _load_from_file(self, file):
        self.code_arr = struct.unpack("i", file.read(4))[0]
        self.mapping_buf = file.read()

    def dump(self, file):
        file.write(struct.pack("i", len(self.code_arr)))
        file.write(self.mapping_buf)

    def fold_or_die(self, input_buf):
        output_buf = bytearray()
        for char in input_buf:
            code = ord(char)
            if mapping := self._find_mapping(code):
                output_buf.extend(self.mapping_buf[mapping[1] : mapping[2]])
            else:
                output_buf.append(code)
        return output_buf.decode("utf-8")

    def _find_mapping(self, code):
        return next((mapping for mapping in self.code_arr if mapping[0] == code), None)
