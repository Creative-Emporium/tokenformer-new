import glob
import configparser

# Load all of the configurations
configs = {}
for f in glob.glob('configs/*.ini'):
    parser = configparser.ConfigParser()
    parser.read(f)
    
    prefix = f[8:-4]

    for s in parser.sections():
        values = {}
        
        for k in parser[s]:
            x = parser[s][k]
            if k == 'training':
                x = x.split(';')
                x = [ int(w.strip()) for w in x ]
                x = {
                    'epochs': x[0],
                    'steps': x[1],
                    'batch': x[2]
                }
            elif k == 'languages':
                x = x.split(';')
                x = [ w.strip() for w in x ]
            elif k == 'grow':
                x = x.split(';')
                x = [ int(w.strip()) for w in x ]
                x = {
                    'amount': x[0],
                    'frequency': x[1]
                }
            elif k in [ 'heads', 'context', 'embedding', 'layers', 'extension' ]:
                x = int(x)
            elif k in [ 'masked', 'incremental' ]:
                x = True if x == 'yes' else False
                
            if x == 'no':
                x = False
            elif x == 'yes':
                x = True
            
            values[k] = x
            
        configs[prefix + '/' + s] = values

# Resolve uses
for s in configs:
    m = None
    
    for k in configs[s]:
        if k == 'use':
            x = configs[s][k]
            m = configs[x]
    
    if not m is None:
        for k in m:
            if not k in configs[s]:
                configs[s][k] = m[k]
                
        del configs[s]['use']