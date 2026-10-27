from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

# GPT-2 style pretokenization pattern (CS336 / assignment handout).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = re.compile(PAT)


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab) if vocab is not None else {}
        self.merges: list[tuple[bytes, bytes]] = list(merges) if merges is not None else []
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        self._rebuild_tables()

    def _rebuild_tables(self) -> None:
        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self._special_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            token_id = self._bytes_to_id.get(token_bytes)
            if token_id is not None:
                self._special_to_id[token] = token_id
        # Longest special tokens first for overlapping matches.
        self._special_sorted = sorted(self.special_tokens, key=len, reverse=True)
        if self._special_sorted:
            pattern = "|".join(re.escape(s) for s in self._special_sorted)
            self._special_re = re.compile(f"({pattern})")
        else:
            self._special_re = None

    def train(
        self, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> None:
        if vocab_size < 256 + len(special_tokens):
            raise ValueError(
                "vocab_size must be at least 256 + number of special tokens"
            )

        self.special_tokens = list(special_tokens)
        # Special tokens first, then single bytes, then merges (matches common CS336 layout).
        vocab: dict[int, bytes] = {}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")
        for i in range(256):
            vocab[len(vocab)] = bytes([i])

        word_freqs = self._count_pre_tokens(input_path, special_tokens)
        # Represent each pre-token as a tuple of single-byte tokens.
        words: dict[tuple[bytes, ...], int] = {
            tuple(bytes([b]) for b in word): freq for word, freq in word_freqs.items()
        }

        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        for word, freq in words.items():
            self._add_word_pairs(word, freq, pair_counts, pair_to_words)

        merges: list[tuple[bytes, bytes]] = []
        num_merges = vocab_size - len(vocab)
        for _ in range(num_merges):
            if not pair_counts:
                break
            # Highest frequency; ties -> lexicographically largest pair.
            best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if pair_counts[best_pair] <= 0:
                break

            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = new_token

            affected = [w for w in pair_to_words.get(best_pair, ()) if w in words]
            new_freqs: Counter[tuple[bytes, ...]] = Counter()
            for old_word in affected:
                freq = words.pop(old_word)
                self._remove_word_pairs(old_word, freq, pair_counts, pair_to_words)
                new_freqs[self._merge_word(old_word, best_pair)] += freq

            for new_word, freq in new_freqs.items():
                words[new_word] = words.get(new_word, 0) + freq
                self._add_word_pairs(new_word, freq, pair_counts, pair_to_words)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

        self.vocab = vocab
        self.merges = merges
        self._rebuild_tables()

    @staticmethod
    def _count_pre_tokens(
        input_path: str, special_tokens: list[str]
    ) -> Counter[bytes]:
        counts: Counter[bytes] = Counter()
        special_set = set(special_tokens)
        if special_tokens:
            pattern = "|".join(
                re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)
            )
            splitter = re.compile(pattern)
        else:
            splitter = None

        with open(input_path, encoding="utf-8", newline="") as f:
            text = f.read()

        chunks = splitter.split(text) if splitter is not None else [text]
        for chunk in chunks:
            if not chunk or chunk in special_set:
                continue
            for match in _PAT.finditer(chunk):
                counts[match.group().encode("utf-8")] += 1
        return counts

    @staticmethod
    def _add_word_pairs(
        word: tuple[bytes, ...],
        freq: int,
        pair_counts: Counter[tuple[bytes, bytes]],
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    ) -> None:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    @staticmethod
    def _remove_word_pairs(
        word: tuple[bytes, ...],
        freq: int,
        pair_counts: Counter[tuple[bytes, bytes]],
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    ) -> None:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            words_with_pair = pair_to_words.get(pair)
            if words_with_pair is not None:
                words_with_pair.discard(word)
                if not words_with_pair:
                    del pair_to_words[pair]

    @staticmethod
    def _merge_word(
        word: tuple[bytes, ...], pair: tuple[bytes, bytes]
    ) -> tuple[bytes, ...]:
        left, right = pair
        merged = left + right
        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for is_special, part in self._split_special(text):
            if not part:
                continue
            if is_special:
                ids.append(self._special_to_id[part])
            else:
                for match in _PAT.finditer(part):
                    ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            ids, buffer = self._encode_complete_prefix(buffer)
            yield from ids
        if buffer:
            yield from self.encode(buffer)

    def _encode_complete_prefix(self, text: str) -> tuple[list[int], str]:
        """Encode a safe prefix; leave a remainder that may still grow."""
        if not text:
            return [], ""

        keep_special = self._trailing_special_prefix_len(text)
        if keep_special:
            text, special_remainder = text[:-keep_special], text[-keep_special:]
        else:
            special_remainder = ""

        parts = list(self._split_special(text))
        if not parts:
            return [], special_remainder

        ids: list[int] = []
        for is_special, part in parts[:-1]:
            if not part:
                continue
            if is_special:
                ids.append(self._special_to_id[part])
            else:
                for match in _PAT.finditer(part):
                    ids.extend(self._encode_pretoken(match.group().encode("utf-8")))

        is_special, part = parts[-1]
        if is_special:
            if part:
                ids.append(self._special_to_id[part])
            return ids, special_remainder

        matches = list(_PAT.finditer(part))
        if not matches:
            return ids, part + special_remainder
        for match in matches[:-1]:
            ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids, matches[-1].group() + special_remainder

    def _trailing_special_prefix_len(self, text: str) -> int:
        max_keep = 0
        for token in self.special_tokens:
            # Proper prefixes only; a full special token is already complete.
            for length in range(1, len(token)):
                if text.endswith(token[:length]):
                    max_keep = max(max_keep, length)
        return max_keep

    def _split_special(self, text: str) -> list[tuple[bool, str]]:
        if self._special_re is None:
            return [(False, text)]
        parts: list[tuple[bool, str]] = []
        special_set = set(self.special_tokens)
        for part in self._special_re.split(text):
            if not part:
                continue
            parts.append((part in special_set, part))
        return parts

    def _encode_pretoken(self, token_bytes: bytes) -> list[int]:
        if not token_bytes:
            return []
        pieces: list[bytes] = [bytes([b]) for b in token_bytes]
        ranks = self._merge_ranks
        while len(pieces) >= 2:
            best_i = -1
            best_rank = None
            for i in range(len(pieces) - 1):
                rank = ranks.get((pieces[i], pieces[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            pieces = (
                pieces[:best_i]
                + [pieces[best_i] + pieces[best_i + 1]]
                + pieces[best_i + 2 :]
            )
        return [self._bytes_to_id[piece] for piece in pieces]

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path_prefix: str) -> None:
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        vocab_path = Path(f"{path_prefix}.vocab.json")
        merges_path = Path(f"{path_prefix}.merges.json")
        config_path = Path(f"{path_prefix}.config.json")

        vocab_hex = {str(i): token.hex() for i, token in self.vocab.items()}
        merges_hex = [[left.hex(), right.hex()] for left, right in self.merges]
        config = {"special_tokens": self.special_tokens, "byte_encoding": "hex"}

        vocab_path.write_text(json.dumps(vocab_hex, ensure_ascii=False), encoding="utf-8")
        merges_path.write_text(json.dumps(merges_hex, ensure_ascii=False), encoding="utf-8")
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    def load(self, path_prefix: str) -> None:
        vocab_path = Path(f"{path_prefix}.vocab.json")
        merges_path = Path(f"{path_prefix}.merges.json")
        config_path = Path(f"{path_prefix}.config.json")

        vocab_hex = json.loads(vocab_path.read_text(encoding="utf-8"))
        merges_hex = json.loads(merges_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.vocab = {int(i): bytes.fromhex(h) for i, h in vocab_hex.items()}
        self.merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]
        self.special_tokens = list(config.get("special_tokens", []))
        self._rebuild_tables()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> BPE:
        vocab_hex = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_hex = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        vocab = {int(i): bytes.fromhex(h) for i, h in vocab_hex.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
