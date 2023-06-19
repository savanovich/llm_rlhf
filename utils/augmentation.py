import augmenty
import nlpaug.augmenter.word as naw
import spacy

from utils.markdown import apply_transformation_for_non_code


nlp = spacy.load("en_core_web_md")
keystroke_error_augmenter = augmenty.load("keystroke_error_v1", level=0.01)
# char_swap_augmenter = augmenty.load("char_swap_v1", level=0.01)
duplicate_token_augmenter = augmenty.load("duplicate_token_v1", level=0.01)

combined_augmenter = augmenty.combine([
    keystroke_error_augmenter,
    # char_swap_augmenter,
    duplicate_token_augmenter])


# back_translation_aug = naw.BackTranslationAug(
#     from_model_name='Helsinki-NLP/opus-mt-en-de',
#     to_model_name='Helsinki-NLP/opus-mt-de-en')


def augment(text):
    return apply_transformation_for_non_code(text, functions=[
        lambda s: list(augmenty.texts([s, ], augmenter=combined_augmenter, nlp=nlp))[0],
        # lambda s: back_translation_aug.augment(s)[0],
    ])
