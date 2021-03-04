import pandas as pd
import os
import random
import json
"""
dataset = {'train': [[{'personality': [...], 'utterances': [{'candidates': [...], 
	'history': [...]}, {'candidates': [...], 'history': [...]}]}]], 
	'valid': ...}


    Procedure:
    personality sentences - may need to write them by hand, however only need around 5.
    utterances: 7 dicts, each with 20 candidate sentences and anywhere between 1 - 13 history sentences.

    removing links:
    import re
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

"""
celebs = "./celeb_data"
dialog_dataset = {'train': [], 'valid': []}
PERSONA_DICT = {
                'ArianaGrande_tweets': ['I love my fans so much', 'dancing in high heals is tough',
                'life is beautiful', 'Music is my biggest passion'], 

                'KimKardashian_tweets': ['I am Armenian, so of course I am obsessed with laser hair removal.', 
                'I have never used my pool', 'when i gain a pound its in the headlines', "i don't talk about money"],

                'KylieJenner_tweets': ['I take, like, 500 selfies to get one I like', "I just don't like the airport. It scares me", 
                "My oldest sister is bossy, my brother is a stirrer and me- we’ll I’m perfect.", "I don’t really regret anything."], 

                'realDonaldTrump_tweets': ['I will have Mexico pay for that wall.', 'I won the election, by a lot', 'Diet coke is my favourite drink', 
                'Make America great again', 'Man, we could use a big fat dose of global warming'], 

                'shanedawson_tweets': ['I want to know the psychology of a YouTuber', 'I started making videos when I was, like, seven or eight.',
                'Anybody could be a sociopath. Anybody.', 'I know what I do for a living is ridiculous.', 
                'There is nothing creepier to me than a student who hangs out with the teachers.'],

                'Zendaya_tweets': ["I'm a Virgo, and I know what I like.", 'I use my platform as a tool and a way to speak about greater change.',
                'I love singing and would love to record an album at some point.', "I've always grown up around theater.", 
                "I'm a chocolate addict."],
                
                'justinbieber_tweets': ["Canada's the best country in the world.", "I've never made a bad song.", 
                'Never say never.', 'Stay humble.']
                }
HISTORY_APPENDS = 2
CANDIDATES_LEN = 20
NUMBER_OF_DICTS = 7

# Loop through files
for file in sorted(os.listdir(celebs)):
    print("Working on", file)
    this_celeb = file[:len(file)-4]
    file = celebs + '/' + file
    
    # testing reading in tweets
    df = pd.read_csv(file)
    data = df.to_dict('records')
    random.shuffle(data)

    # testing building dataset
    history = ['hi! how are you doing?']
    utterances = []

    for i in range(NUMBER_OF_DICTS):
        this_candidates = []
        this_history = []
        this_utterance = {'candidates': [], 'history': []}
        for t in range(CANDIDATES_LEN):
            this_candidates.append(data[t]['text'])
        
        data = data[CANDIDATES_LEN:]

        if i > 0:
            for i in range(HISTORY_APPENDS):
                this_history.append(data[i]['text'])
        
            data = data[HISTORY_APPENDS:]
        
        for it, x in enumerate(history):
            this_history.insert(it, x)
        
        history = this_history

        this_utterance['candidates'] = this_candidates
        this_utterance['history'] = this_history
        utterances.append(this_utterance)

    celeb_persona = PERSONA_DICT[this_celeb]
    this_persona = {'personality': celeb_persona, 'utterances': utterances}
    dialog_dataset['train'].append(this_persona)
    dialog_dataset['valid'].append(this_persona)


with open("celebs_dialog_dataset.json", 'w', encoding='utf-8') as f:
    json.dump(dialog_dataset, f, ensure_ascii=False)

print("Done.")