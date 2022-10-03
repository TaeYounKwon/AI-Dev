letter=['a','b','c','d','e']
for i in range(len(letter)):
    letter_start = letter[i]
    print('letter: ',letter[i])
    letter[i]=letter[i]+'x'
    print('letter changes x : ',letter[i])
    print('letter start: ',letter_start)
    if(letter[i]=='c'):
        break
