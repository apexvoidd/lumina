import re
import json
import os

class CustomTokenizer:
    def __init__(self):
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<user>": 2,
            "<ai>": 3,
            "<end>": 4
        }
        self.vocab = dict(self.special_tokens)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def _tokenize_text(self, text):
        # Normalize text to lowercase and split words/punctuation
        text = text.lower()
        # Regex to find words or single punctuation characters
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocab(self, corpus_text):
        tokens = self._tokenize_text(corpus_text)
        
        # Count token frequencies
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
            
        # Add words to vocabulary, starting index after special tokens
        idx = len(self.special_tokens)
        # Sort by frequency to make it deterministic and structured
        sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for token, count in sorted_tokens:
            if token not in self.vocab:
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
                idx += 1
                
        print(f"Vocabulary built. Total tokens: {len(self.vocab)}")

    def encode(self, text, add_special_tokens=False, is_user=True):
        raw_tokens = self._tokenize_text(text)
        
        ids = []
        if add_special_tokens:
            ids.append(self.vocab["<user>" if is_user else "<ai>"])
            
        for token in raw_tokens:
            ids.append(self.vocab.get(token, self.vocab["<unk>"]))
            
        if add_special_tokens:
            ids.append(self.vocab["<end>"])
            
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if isinstance(idx, int):
                token_val = self.inv_vocab.get(idx, "<unk>")
            else:
                token_val = self.inv_vocab.get(idx.item(), "<unk>")
            
            # Skip control/pad tokens in final text output, except <end> which we can use to stop
            if token_val not in ["<pad>", "<user>", "<ai>"]:
                tokens.append(token_val)
                
        # Reconstruct sentence with basic spacing rules
        text = ""
        for i, t in enumerate(tokens):
            if t == "<end>":
                break
            if i > 0 and t not in [".", ",", "?", "!", "'", "s", "m", "re", "t"]:
                # Don't add space before punctuation or common contractions
                text += " "
            text += t
            
        # Clean up some common contraction spacings, e.g. "it ' s" -> "it's"
        text = text.replace(" ' s", "'s").replace(" ' t", "'t").replace(" ' m", "'m").replace(" ' re", "'re").replace(" ' ve", "'ve")
        return text

    def save(self, filepath):
        data = {
            "vocab": self.vocab,
            "inv_vocab": {str(k): v for k, v in self.inv_vocab.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        # Convert keys back to integers for inv_vocab
        self.inv_vocab = {int(k): v for k, v in data["inv_vocab"].items()}
        return True

    @property
    def vocab_size(self):
        return len(self.vocab)
