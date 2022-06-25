import requests
fname = 'data.mat'
url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
r = requests.get(url)
open(fname , 'wb').write(r.content)