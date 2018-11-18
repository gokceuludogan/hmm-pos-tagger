import re 

def is_verb(word):
    word = word.lower()
    if re.search(r'(yor)(du(k|m|n)?|sun(uz)?|sa|muş|u(m|z))?(lar)?\b', word):
        return True
    elif re.search(r'(m[iıuü]ş)(t[iıuü](m|k|n[iıuü]z|ys[ae])?|[iıuü](m|z)|s[iıuü]n|t[iıuü]|s[ae](l[ae]r)?)?(l[ae]r(s[ae])?)?\b', word):
        return True
    elif re.search(r'(d[iıuü]|t[iıuü])\b', word):
        return True        
    elif re.search(r'(d[iıuü]|t[iıuü])(m|n([iıuü]z)?|k|l[ae]r)?\b', word):
        return True
    elif re.search(r'(r)(l[ae]r)?\b', word):
        return True     
    elif re.search(r'(bilir)(i(m|z)?|sin(iz)?|ler)?\b', word):
        return True     
    elif re.search(r'(s[ae])(m|n([ıi]z)?|k|l[ae]r)?\b', word):
        return True 
    elif re.search(r'(s[iıuü]n([iıuü]z)?)\b', word):
        return True             
    elif re.search(r'(m[ae]l[iı])(y[iı](m|z)|s[iı]n([iı]z)?|l[ae]r)?\b', word):
        return True
    elif re.search(r'([ae]c[ae]k)(l[ae]r)?\b', word):
        return True
    elif re.search(r'([ae]c[ae]ğ[iı](m|z))\b', word):
        return True 
    elif re.search(r'(y[iıuü](m|z))\b', word):
        return True
    elif re.search(r'(m[ae]z)(l[ae]r)?\b', word):
        return True
    elif re.search(r'([ae]l[iı]m)\b', word):
        return wwwwwwwwwwwww/.
    elif re.search(r'(r[iıuü]m)\b', word):
        return True
    elif re.search(r'([iı]z)\b', word):
        return True
    elif re.search(r'(m[ae]kt[ae])\b', word):
        return True                                     
    elif re.search(r'(m[ae](m|y[iı]n))\b', word):
        return True
    elif re.search(r'((y)?([iıuü]n))\b', word):
        return True                    opppppppppppppppppp0000000000000w                 
    else:
        return False


def is_noun(word):
    word = word.lower()
    if re.search(r'(n[iıuü]n)\b', word):
        return True
    elif re.search(r'((n|y|m)[ae])\b', word):
        return True
    elif re.search(r'([dt][ae](n)?)\b', word):
        return True 
    elif re.search(r'((y)?l[ae])\b', word):
        return True 
    elif re.search(r'(l[ae]r([ae]|[iı]([mn]([iı]z)?)?|l[ae]|[iı](n([iı]|d[ea])?)?)?)\b', word):
        return True 
    elif re.search(r'(s[iıuü])(n([iıuü])?)?\b', word):
        return True         
    elif re.search(r'(\'[aeiıuü])\b', word):
        return True 
    elif re.search(r'(m[ae]k)\b', word):
        return True 
    else:
        return False        

def is_ques(word):
    word = word.lower()
    if re.search(r'(m[iıuü](ym[iıuü]ş|y[iıuü]m)?)\b', word):
        return True
    else:
        return False

def is_adv(word):
    word = word.lower()
    if re.search(r'([ıiuü]nc[ae])\b', word):
        return True
    elif re.search(r'([ae]r[ae]k)\b', word):
        return True 
    elif re.search(r'(([ıiuü]k)?ç[ae])\b', word):
        return True
    elif re.search(r'((n)?[ıiuü]p)\b', word):
        return True             
    elif re.search(r'((m[ae])?d[ae]n)\b', word):
        return True     
    elif re.search(r'(ken)\b', word):
        return True 
    elif re.search(r'([ae]l[ıi])\b', word):
        return True         
    else:
        return False

def is_adj(word):
    word = word.lower()
    if re.search(r'(s[ıiuü]z)\b', word):
        return True
    elif re.search(r'(l[ıiuü](k)?)\b', word):
        return True 
    elif re.search(r'([dt][ıiuü]ğ[ıiuü])\b', word):
        return True     
    elif re.search(r'([çc][ıiaeuü])\b', word):
        return True     
    elif re.search(r'([ae][nl])\b', word):
        return True     
    elif re.search(r'(([dt][ae])?ki)\b', word):
        return True     
    elif re.search(r'([ıiuü]ğ[ıiuü][mn]([ıiuü]z)?)\b', word):
        return True 
    elif re.search(r'(m[ıiuü]ş)\b', word):
        return True 
    elif re.search(r'([ae]c[ae]k)\b', word):
        return True         
    elif re.search(r'([ıiuü]ms[ıiuü])\b', word):
        return True 
    elif re.search(r'([ıiuü]k)\b', word):
        return True
    elif re.search(r'([ae]c[ae]ğ[ıi])\b', word):
        return True 
    elif re.search(r'([gk][aeıiuü]n)\b', word):
        return True
    elif re.search(r'(tik|tif|sal)\b', word):
        return True                 
    else:
        return False

def is_pron(word):
    word = word.lower()
    if re.search(r'\b(biz|ben|siz|onlar|sen|kendi|hep|hiçbir|kimi|birbir|bu|şu|kimi|bazı|o(n(a|u(n)?))?\b)', word):
        return True
    elif re.search(r'(ki(ler|ne)?)\b', word):
        return True
    else:
        return False

def is_det(word):
    word = word.lower()
    if re.search(r'(bazı|bir(çok|kaç)?|bu|her|hiçbir|kimi|o|şu|tüm)\b', word):
        return True
    elif re.search(r'(d[ae]ki)\b', word):
        return True
    else:
        return False