import _pickle as pkl
import zlib


PATH = 'Expert Data/Complete_Expert_Data.pkl'
with open('Expert Data/Expert_States.pkl', 'rb') as handle:
    ex_data = pkl.load(handle)

comp_states = []
for state in ex_data:
    c_state = zlib.compress(pkl.dumps(state))
    comp_states.append(c_state)

with open('Expert Data/Compressed_Expert_States.pkl', 'wb') as handle:
    pkl.dump(comp_states, handle)