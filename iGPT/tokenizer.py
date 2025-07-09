class Tokenizer:
    def __init__(self, texts, offset):
        self.texts = texts
        self.offset = offset
        self.all_words = set()
        for text in texts:
            self.all_words.update(text.split())
        
        # Convert set to list for consistent ordering
        self.all_words = list(self.all_words)
        self.vocab_size = len(self.all_words)

        # Add special tokens after calculating vocab_size
        # Reserve token 0 for BOS token
        self.bos_token = 0
        self.end_of_text_token = self.vocab_size + self.offset
        self.end_of_image_token = self.vocab_size + 1 + self.offset
        self.all_words.extend(['<end_of_text>', '<end_of_image>'])
        
        # Create mappings with offset applied (starting from 1 to reserve 0 for BOS)
        self.word_to_id = {word: i + 1 + self.offset for i, word in enumerate(self.all_words)}
        self.id_to_word = {i + 1 + self.offset: word for i, word in enumerate(self.all_words)}
        # Add BOS token to mappings
        self.id_to_word[self.bos_token] = '<bos>'

        
    def text_encode(self, text):
        tokens = [self.word_to_id[word] for word in text.split()]
        return torch.tensor(tokens)
        
    def text_decode(self, tokens):
        return ' '.join([self.id_to_word[token] for token in tokens if token != self.end_of_text_token and token != self.bos_token])