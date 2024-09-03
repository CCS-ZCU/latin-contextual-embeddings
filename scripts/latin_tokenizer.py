class LatinTokenizer:
    def __init__(self, vocab_file, encoder, **kwargs):
        self.encoder = encoder
        self.vocab = {}
        self.reverse_vocab = {}
        self._load_vocab(vocab_file)

    def _load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab.get('[UNK]', 1)) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.reverse_vocab.get(i, '[UNK]') for i in ids]

    def tokenize(self, text):
        tokens = text.split()
        wp_tokens = []
        for token in tokens:
            if token in self.vocab:
                wp_tokens.append(token)
            else:
                wp_toks = self.encoder.encode(token)
                wp_tokens.extend([self.reverse_vocab.get(wp + 5, '[UNK]') for wp in wp_toks])
        return wp_tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, return_offsets_mapping=False):
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Create output dictionary
        output = {
            'input_ids': input_ids,
            'tokens': tokens
        }

        if return_tensors == 'pt':
            import torch
            # Convert to tensor and handle padding/truncation
            input_ids_tensor = torch.tensor([input_ids])
            if padding:
                # Handle padding
                pad_length = max_length - len(input_ids) if max_length else 0
                if pad_length > 0:
                    input_ids_tensor = torch.cat([input_ids_tensor, torch.zeros((1, pad_length), dtype=torch.long)], dim=1)
            if truncation and max_length and len(input_ids) > max_length:
                input_ids_tensor = input_ids_tensor[:, :max_length]

            output['input_ids'] = input_ids_tensor

        if return_offsets_mapping:
            # Simulate offsets mapping (just placeholder logic)
            offsets = [(0, len(text)) for _ in tokens]
            output['offset_mapping'] = offsets

        return output