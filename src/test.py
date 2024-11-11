from gamegrid import makeGameGrid, show, doRun, addActor, Location, getOneActorAt, getNumberOfActors, getActors, GGMouse, Actor, toLocationInGrid, toPoint, Color
 
MAX_ACTORS = 18
 
 
def showGameGrid():
    makeGameGrid(11, 10, 70, Color.red, False, mousePressed=pressCallback,
                 mouseDragged=dragCallback, mouseReleased=releaseCallback)
    show()
 
def hideStartScreen():
    showGameGrid()
    doRun()
 
def mouseTouched(actor, mouse, spot):
    hideStartScreen()
 
def showStartScreen():
    makeGameGrid(10, 10, 100, Color.white)
 
    background_actor = Actor("C:\\Users\\Admin\\Downloads\\hintergrund.png")
    addActor(background_actor, Location(5, 2))
 
    start_button = Actor("C:\\Users\\Admin\\Downloads\\Cartoon-start-button-icon-on-transparent-background-PNG.png")
    addActor(start_button, Location(4, 3))
 
    start_button.addMouseTouchListener(mouseTouched, GGMouse.lPress)
    doRun()
    show()
 
x_list = [1, 2, 5, 6]
y_list = [3, 4, 7, 8]
# ...
def Feld(loc):
    if loc.x in x_list or loc.y in y_list:
        return False
    return True
actors = []  # Liste zum Verfolgen aller Steine
selected_actor = None  # Der ausgewählte Stein
def pressCallback(e):
    global actors, selected_actor
    startLoc = toLocationInGrid(e.getX(), e.getY())
    # Überprüfen Sie, ob das Feld erlaubt ist
    if Feld(startLoc):
        actorAtStartLoc = getOneActorAt(startLoc)
        if actorAtStartLoc is not None:
            selected_actor = actorAtStartLoc
        else:
            # Überprüfen Sie, ob die maximale Anzahl von Actors noch nicht erreicht ist
            if getNumberOfActors() < MAX_ACTORS:
                if getNumberOfActors() % 2 == 1:
                    newActor = Actor("sprites/marble.png")  # Spieler 1
                else:
                    newActor = Actor("sprites/token_1.png")  # Spieler 2
                addActor(newActor, startLoc)
                actors.append(newActor)
def dragCallback(e):
    global selected_actor
    if selected_actor is not None:
        startPoint = toPoint(selected_actor.getLocation())
        selected_actor.setLocationOffset(e.getX() - startPoint.x, e.getY() - startPoint.y)
def isOccupied(loc):
    return getOneActorAt(loc) is not None
def releaseCallback(e):
    global selected_actor
    if selected_actor is not None:
        destLoc = toLocationInGrid(e.getX(), e.getY())
        # Überprüfen Sie, ob das Ziel-Feld erlaubt und nicht besetzt ist
        if Feld(destLoc) and not isOccupied(destLoc):
            startPoint = toPoint(selected_actor.getLocation())
            selected_actor.setLocationOffset(0, 0)
            selected_actor.setLocation(destLoc)
        else:
            startPoint = toPoint(selected_actor.getLocation())
            selected_actor.setLocationOffset(0, 0)  # Setzen Sie den Stein zurück an die Startposition
        selected_actor = None  # Setzen Sie den ausgewählten Stein zurück

 
#        Regel()
 
# Regeln-Funktion
#def Regel():
#    red_count = 0
#    blue_count = 0
#
#    for actor in getActors():
#        if actor.getImagePath().endswith("marble.png"):
#            red_count += 1
#        elif actor.getImagePath().endswith("token_1.png"):
#            blue_count += 1
#
#    if red_count < 2 or blue_count < 2:
#        setTitle("Good Game")
 
showStartScreen()