import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos

# constable, lionel, va, boudin, watts, lee, cox

token1 = np.load("tokens/constable.npy")
token2 = np.load("tokens/lionel.npy")

token_mean1 = np.mean(token1, axis=0).reshape(1,-1)
token_mean2 = np.mean(token2, axis=0).reshape(1,-1)

token_sim = cos(token_mean1, token_mean2)

eu = np.zeros(8)

print(token1.shape)

for i in range(8):
    eu[i] = np.sqrt(np.sum(np.square(token1[i,:] - token2[i,:])))


print(token_sim)
print(eu)

##############################################################################################################################################

############################################################# CONTROLNET #####################################################################

##############################################################################################################################################

# constable - constable: 1.
# constable - va: 0.02224347
# constable - boudin: -0.05858683
# constable - lee: -0.04835781
# constable - lionel: 0.0007894
# constable - watts: -0.04274431
# constable - cox: -0.01996468

###############################################################################################################################################

############################################################# STABLE DIFFUSION ################################################################

###############################################################################################################################################


######## gray-sky paintings constable #########

# constable - constable: 0.9999998
# constable - va: 0.97141457
# constable - boudin: 0.96713996
# constable - lee: 0.97024393
# constable - lionel: 0.9734396
# constable - watts: 0.9732511

######## sky paintings #########

# constable - constable: 1.0000004
# constable - lionel: 0.97206795
# constable - va: 0.94715333
# constable - boudin: 0.96267784
# constable - watts: 0.95839983
# constable - cox: 0.93659633
# constable - lee: 0.9390787

### Next step - bootstrap constable-lionel, constable-va - paired t test

# artists are not Realist - around rlaism - va - before reliasm label - monet - not relaist - ifuence of realist painters
# what is reliats io sless useful - care more about period eye 
# exhibitions - truthful - how it aigns with automatic  results - like that idea 

######## old results #########

# constable - constable: 1.0000002
# constable - lionel: 0.96817565
# constable - va: 0.95588243
# constable - boudin: 0.95225775

##### New artists ######
# constable - watts: 0.96759415
# constable - cox: 0.94594586
# constable - lee: 0.95053077

# constable - gogh: 0.93010426
# constable - landscape: 0.91335917
# constable - sketch: 0.9056036
# constable - images: 0.898628
                                    
# gogh - landscape: 0.8923956

# va - landscape: 0.91201454

# new - gogh (landscape paintings), landscapes, images from imagenet, sketches, new painters (watts, cox, lee)
# TODO: picasso


