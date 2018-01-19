import shelve

shelf = {}
def init_shelf():
    try:
        shelf['0'] = 'Unknown'
        shelf['1'] = 'Stanley'
        shelf['2'] = 'Justin'
        shelf['3'] = 'Yasir'
        shelf['4'] = 'Jo'
        shelf['5'] = 'Nisarg'
    finally:
        shelf.close()

def test_shelf():
    print shelf

def clean_shelf():
    for key, val in shelf.iteritems():
        if "kojima" in val:
            del shelf[key]

if __name__ == '__main__':
    shelf = shelve.open('model/user_shelf.db')
    #init_shelf()
    test_shelf()
    #clean_shelf()
    