##################################################################
###################  Object Oriented Python ######################
##################################################################

class PartyAnimal:
    x = 0
    name = ""

    def __init__(self, nam):
        self.name = nam
        print(self.name,'constructed')

    def party(self):
        self.x = self.x + 1
        print(self.name,"party count", self.x)

an = PartyAnimal('Joseph') #Construction

#an.party() # Method invokuing

##### Inheritance ####
class FootballFan(PartyAnimal): #Has every capability thta PartyAnimal has
    points = 0

    def touchdown(self):
        self.points = self.points + 7
        self.party()
        print(self.name,'points',self.points)

s = PartyAnimal('Sally')
s.party()

j = FootballFan('Jim')
j.party()
j.touchdown()












print()
