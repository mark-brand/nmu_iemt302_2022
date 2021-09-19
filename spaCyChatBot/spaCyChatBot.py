import logging
import os
import sys
import random
import re
import spacy
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
load_dotenv()

class SentenceTyper(spacy.matcher.Matcher):
    """Derived matcher meant for determining the sentence type"""
    def __init__(self, vocab):
        super().__init__(vocab)
        # Interrogative (question)
        self.add("WH-QUESTION", [[{"IS_SENT_START": True, "TAG": {"IN": ["WDT", "WP", "WP$", "WRB"]}}]])
        self.add("YN-QUESTION",
                 [[{"IS_SENT_START": True, "TAG": "MD"}, {"POS": {"IN": ["PRON", "PROPN", "DET"]}}],
                  [{"IS_SENT_START": True, "POS": "VERB"}, {"POS": {"IN": ["PRON", "PROPN", "DET"]}}, {"POS": "VERB"}]])
        # Imperative (instructions)
        self.add("INSTRUCTION",
                 [[{"IS_SENT_START": True, "TAG": "VB"}],
                  [{"IS_SENT_START": True, "LOWER": {"IN": ["please", "kindly"]}}, {"TAG": "VB"}]])
        # Wish request
        self.add("WISH",
                 [[{"IS_SENT_START": True, "TAG": "PRP"}, {"TAG": "MD"},
                  {"POS": "VERB", "LEMMA": {"IN": ["love", "like", "appreciate"]}}],  # e.g. I'd like...
                  [{"IS_SENT_START": True, "TAG": "PRP"}, {"POS": "VERB", "LEMMA": {"IN": ["want", "need", "require"]}}]])
        # Exclamatory (emotive)
        # Declarative (statements)

    def __call__(self, *args, **kwargs):
        """inspects the first match, and returns the appropriate sentence type handler"""
        matches = super().__call__(*args, **kwargs)
        if matches:
            match_id, _, _ = matches[0]
            if match_id == self.vocab["WH-QUESTION"]:
                return wh_question_handler
            elif match_id == self.vocab["YN-QUESTION"]:
                return yn_question_handler
            elif match_id == self.vocab["WISH"]:
                return wish_handler
            elif match_id == self.vocab["INSTRUCTION"]:
                return instruction_handler
        else:  # either 'cos there's no matches, or we haven't yet got a custom handler
            return generic_handler
        if len(matches) > 1:
            logger.debug(f"NOTE: SentenceTyper actually found {len(matches)} matches.")


class VerbFinder(spacy.matcher.DependencyMatcher):
    """Derived matcher meant for finding verb phrases"""
    def __init__(self, vocab):
        super().__init__(vocab)
        self.add("VERBPHRASE", [
            [{"RIGHT_ID": "root", "RIGHT_ATTRS": {"DEP": "ROOT"}},
             {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "auxiliary", "RIGHT_ATTRS": {"TAG": "VB"}},
             {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "modal", "RIGHT_ATTRS": {"TAG": "MD"}}],
            [{"RIGHT_ID": "root", "RIGHT_ATTRS": {"DEP": "ROOT"}},
             {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "auxiliary", "RIGHT_ATTRS": {"POS": "AUX"}}],
            [{"RIGHT_ID": "root", "RIGHT_ATTRS": {"DEP": "ROOT"}}]
        ])

    def __call__(self, *args, **kwargs):
        """returns the sequence of token ids which constitute the verb phrase"""
        verbmatches = super().__call__(*args, **kwargs)
        if verbmatches:
            if len(verbmatches) > 1:
                logging.debug(f"NOTE: VerbFinder actually found {len(verbmatches)} matches.")
                for verbmatch in verbmatches:
                    logging.debug(verbmatch)
            _, token_idxs = verbmatches[0]
            return sorted(token_idxs)


povs = {
    "I am": "you are",
    "I was": "you were",
    "I'm": "you're",
    "I'd": "you'd",
    "I've": "you've",
    "I'll": "you'll",
    "you are": "I am",
    "you were": "I was",
    "you're": "I'm",
    "you'd": "I'd",
    "you've": "I've",
    "you'll": "I'll",
    "I": "you",
    "my": "your",
    "your": "my",
    "yours": "mine",
    "you": "I",  # as subject, else "me"
    "me": "you",
}
povs_c = re.compile(r'\b({})\b'.format('|'.join(re.escape(pov) for pov in povs)))


def wh_question_handler(nlp, sentence, verbs_idxs):
    """Requires a qualitative answer. For now, very similar to yn_question_handler"""
    logging.debug(f"INVOKING WH-QUESTION HANDLER {verbs_idxs}")
    reply = []
    reply.append(sentence[0].text.lower())  # by definition, the first word is a wh-word
    part = [chunk.text for chunk in sentence.noun_chunks if chunk.root.dep_ == 'nsubj']
    if part:
        reply.append(part[0])
    reply.append(" ".join([sentence[i].text.lower() for i in verbs_idxs]))
    part = [chunk.text for chunk in sentence.noun_chunks if chunk.root.dep_ == 'dobj']
    if part:
        reply.append(part[0])
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), " ".join(reply))
    reply = random.choice(["I don't know ", "I can't say "]) + reply
    reply += random.choice([", but I'll try to find out for you. Please check in with me again later.",
                            ", but perhaps that's something I'd be able to find out for you. Remind me, if I forget.",
                            ". I'll see if I can find out, though. Ask me again sometime."])
    return reply


def yn_question_handler(nlp, sentence, verbs_idxs):
    """Requires a binary answer. For now, very similar to wh_question_handler"""
    logging.debug("INVOKING YN-QUESTION HANDLER")
    reply = []
    part = [chunk.text for chunk in sentence.noun_chunks if chunk.root.dep_ == 'nsubj']
    if part:
        reply.append(part[0])
    reply.append(" ".join([sentence[i].text.lower() for i in verbs_idxs]))
    part = [chunk.text for chunk in sentence.noun_chunks if chunk.root.dep_ == 'dobj']
    if part:
        reply.append(part[0])
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), " ".join(reply))
    reply = random.choice([
        "I don't know whether ",
        "I can't say if ",
    ]) + reply
    reply += random.choice([
        " at this very moment. Let me find out.",
        ". I may have to think about this some more.",
    ])
    return reply


def wish_handler(nlp, sentence, verbs_idxs):
    """Expresses a wish"""
    logging.debug("INVOKING WISH HANDLER")
    reply = sentence.text
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), reply)
    reply = random.choice([
        "Understood: ",
        "Got it: ",
    ]) + reply
    reply += random.choice([
        " I'll see what I can do.",
        "",
    ])
    return reply


def instruction_handler(nlp, sentence, verbs_idxs):
    """Requires action"""
    logging.debug("INVOKING INSTRUCTION HANDLER")
    reply = sentence.text
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), reply)
    reply = random.choice([
        "Understood: ",
        "Got it: ",
    ]) + reply
    reply += random.choice([
        " What do you think about that?",
        " Thanks for sharing.",
    ])
    return reply


def generic_handler(nlp, sentence, verbs_idxs):
    """Requires something else"""
    logging.debug("INVOKING GENERIC HANDLER")
    reply = sentence.text
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), reply)
    return reply


def banter(update, context):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(update.message.text)
    sentencetyper = SentenceTyper(nlp.vocab)
    verbfinder = VerbFinder(nlp.vocab)

    reply = ''
    for sentence in doc.sents:
        verbs_idxs = verbfinder(sentence.as_doc())
        reply += (sentencetyper(sentence.as_doc()))(nlp, sentence, verbs_idxs)

    update.message.reply_text(reply)
    return


def start(update, context):
    """announce yourself in a way that suggests the kind of interaction expected"""
    update.message.reply_text("Hi! I am your bot. How may I be of service?")
    return 'BANTER'


def cancel(update, context):
    """gracefully exit the conversation"""
    update.message.reply_text("Thanks for the chat. I'll be off then!")
    return ConversationHandler.END


def help(update, context):
    """what situations give rise to a request such as this?"""
    update.message.reply_text("I strongly suggest that you read the manual.")
    return


def main():
    """the bot's main message loop is set up and run from here"""
    updater = Updater(os.getenv('API_TOKEN'))
    dispatch = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler(['start', 'order'], start)],
        states={
            # a dict of states needs to be inserted here
            'BANTER': [MessageHandler(Filters.text & ~Filters.command, banter)],
        },
        fallbacks=[CommandHandler(['cancel', 'stop', 'exit'], cancel),
                   CommandHandler('help', help)]
    )
    dispatch.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
