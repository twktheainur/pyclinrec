import ctypes
import os

from pyclinrec.recognizer.mgrep.trienode import TreeNode


class Trie:
    kBufferIncrement = 1048576

    def __init__(self, filename, casefolding):
        self.root_node = None
        self.casefolding = casefolding
        try:
            with open(filename, "r", encoding="utf-8") as fdict:
                line_num = 0
                for line in fdict:
                    line = line.rstrip("\r\n")
                    line_num += 1
                    if not line:
                        continue
                    try:
                        id_str, word = line.split("\t", 1)
                        id_num = int(id_str)
                    except ValueError as e:
                        raise ValueError(
                            f"Dictionary file line {line_num} is improperly formatted."
                        ) from e
                    if self.casefolding:
                        word = self.casefolding.fold_or_die(word)
                    if self.root_node:
                        self.root_node.insert_word(word, id_num)
                    else:
                        self.root_node = TreeNode(word, id_num)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{filename}: No such file") from e

    def annotate(self, word_divider, longest, input_buf):
        output_buf = []
        output_size = 0
        output_tail = 0
        current_ptr = 0
        farthest_matching_ending = -1
        orig_input = input_buf

        if not word_divider.get_total():
            while current_ptr < len(input_buf):
                current_match_ending, output_buf, output_size, output_tail = (
                    self.annotate_recursive(
                        word_divider,
                        longest,
                        input_buf,
                        output_buf,
                        output_size,
                        output_tail,
                        current_ptr,
                        self.root_node,
                        current_ptr,
                        farthest_matching_ending,
                        orig_input,
                    )
                )
                if current_match_ending > farthest_matching_ending:
                    farthest_matching_ending = current_match_ending
                current_ptr += 1  # Simulate ReadUnicodeOrDie
        else:
            previous_is_divider = True
            while current_ptr < len(input_buf):
                next_ptr = current_ptr
                code = ord(input_buf[next_ptr])  # Simulate ReadUnicodeOrDie
                current_is_divider = word_divider.is_divider(code)
                if not current_is_divider and previous_is_divider:
                    current_match_ending, output_buf, output_size, output_tail = (
                        self.annotate_recursive(
                            word_divider,
                            longest,
                            input_buf,
                            output_buf,
                            output_size,
                            output_tail,
                            current_ptr,
                            self.root_node,
                            current_ptr,
                            farthest_matching_ending,
                            orig_input,
                        )
                    )
                    if current_match_ending > farthest_matching_ending:
                        farthest_matching_ending = current_match_ending
                previous_is_divider = current_is_divider
                current_ptr = next_ptr + 1  # Simulate ReadUnicodeOrDie

        return output_buf

    def annotate_recursive(
        self,
        word_divider,
        longest,
        input_buf,
        output_buf,
        output_size,
        output_tail,
        match_starting,
        tree_node,
        current_ptr,
        farthest_matching_ending,
        orig_input,
    ):
        """
        Recursively annotates the input buffer based on the trie structure.

        Args:
            word_divider (WordDivider): The word divider object.
            longest (bool): Flag indicating whether to find the longest match.
            input_buf (str): The input buffer to be annotated.
            output_buf (list): The output buffer to store the annotated results.
            output_size (list): The size of the output buffer.
            output_tail (list): The tail index of the output buffer.
            match_starting (int): The starting index of the current match.
            tree_node (TreeNode): The current node in the trie structure.
            current_ptr (int): The current pointer in the input buffer.
            farthest_matching_ending (int): The ending index of the farthest matching substring.
            orig_input (str): The original input buffer.

        Returns:
            int: The ending index of the current match.
        """

        def should_update_output_buf():
            return (
                tree_node.get_id_size() > 0
                and (
                    not longest
                    or (
                        not current_match_ending
                        and current_ptr > farthest_matching_ending
                    )
                )
                and (
                    not word_divider.get_total()
                    or word_divider.is_boundary(input_buf[current_ptr])
                )
            )

        def update_output_buf():
            for _ in range(tree_node.get_id_size()):
                if (
                    output_size[0] - output_tail[0] - (current_ptr - match_starting)
                    <= 128
                ):
                    output_size[0] += self.kBufferIncrement
                    output_buf[0] = output_buf[0][: output_tail[0]] + (
                        "\0" * (output_size[0] - len(output_buf[0]))
                    )
                len_written = len(
                    output_buf[0][output_tail[0] :]
                )  # Placeholder for actual write operation
                output_tail[0] += len_written
                output_buf[0] = (
                    output_buf[0][: output_tail[0]]
                    + input_buf[match_starting:current_ptr]
                    + "\n"
                )
                output_tail[0] += current_ptr - match_starting + 1

        current_match_ending = None
        original_current_ptr = current_ptr
        word = tree_node.word()

        while (
            current_ptr < len(input_buf)
            and len(word) > 0
            and input_buf[current_ptr] == word[0]
        ):
            current_ptr += 1
            word = word[1:]

        if len(word) == 0:
            if current_ptr < len(input_buf) and tree_node.right():
                current_match_ending = self.annotate_recursive(
                    word_divider,
                    longest,
                    input_buf,
                    output_buf,
                    output_size,
                    output_tail,
                    match_starting,
                    tree_node.right(),
                    current_ptr,
                    farthest_matching_ending,
                    orig_input,
                )
            if should_update_output_buf():
                update_output_buf()
                current_match_ending = current_ptr
        elif len(word) > 0 and word[0] < input_buf[current_ptr] and tree_node.down():
            current_match_ending = self.annotate_recursive(
                word_divider,
                longest,
                input_buf,
                output_buf,
                output_size,
                output_tail,
                match_starting,
                tree_node.down(),
                original_current_ptr,
                farthest_matching_ending,
                orig_input,
            )
        return current_match_ending
