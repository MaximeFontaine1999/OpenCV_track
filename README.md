# OpenCV_track

   Dans le cadre d'un projet de robotique universitaire, il était nécessaire que notre robot ait besoin d'une "vision". Cette "vision" est donc très importante afin que le robot puisse faire certaines actions dont, par exemple, celle de récupérer un objet d'une certaine couleur. Pour cela nous avions besoin que ce robot puisse reconnaitre n'importe quelle forme d'objet de n'importe quelle couleur.
 
Avant de commencer un quelconque programme, je me devais de choisir avec quel outil travailler. Je me suis donc penché vers opencv et son utilisation en python.

Une fois ce choix effectué, j'ai dans un premier temps fait des recherches sur la reconnaissance de couleur sous opencv.

## Reconnaissance de couleur

Après plusieurs recherches, j'ai trouvé un programme (range_detector.py) qui permet de définir un masque RGB ou HSV à partir d'une photo ou directement de la webcam. Lors de mes recherches, j'ai pu observer qu'il était recommandé de travailler avec les masques HSV plutôt qu'en RGB. En effet, comme nous pouvons le voir ci-dessous il est naturellement plus simple de définir un intervalle contenant une unique teinte dans différente intensité en HSV alors qu'en RGB si on veut la même teinte sous différentes intensités on va récupérer des teintes que l'on ne veut pas.  


<img src="https://e7.pngegg.com/pngimages/643/12/png-clipart-rgb-color-model-hsl-and-hsv-rgb-color-space-cube-cube-blue-color.png" width="350">  <img src="https://w0.pngwave.com/png/982/449/rgb-color-model-rgb-color-space-hsl-and-hsv-cube-colors-png-clip-art.png" width="350">

Une fois l'intervalle défini nous pouvons donc travailler sur la reconnaissance d'objet de cette couleur. Pour cela, il faut au préalable définir les valeurs que nous avons obtenues grace au programme précédent.
    
    greenLower = (30, 70, 30)
    greenUpper = (93, 255, 255)
    
Ensuite, il faut transformer l'image en lui appliquant un filtre qui va transformer les couleur RGB en HSV

    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
L'image étant converti en HSV on peut dès à présent appliquer notre masque, c'est-à-dire que nous allons récupérer tout ce qui apparait à l'écran et qui est dans l'intervalle de couleur défini

    mask1 = cv2.inRange(hsv, greenLower, greenUpper)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)
    
L'étape suivante consiste à chercher les contours parmi les objets restants après le masque 

    cntg = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntg = imutils.grab_contours(cntg)
    
Ainsi il ne reste que les objets de la couleur souhaitée.

## Reconnaissance de forme

La deuxième étape consiste donc à reconnaitre la forme de ces objets.
Avant d'identifier la forme des objets de la couleur souhaitée, nous allons le faire sur tous les objets.
Pour cela, la méthode est similaire à la précédente sur les premières étapes, c'est-à-dire qu'à la place de transformer l'image en HSV nous allons la transformer en nuances de gris afin de mieux observer les contours des formes.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
    edged = cv2.Canny(gray, 70, 150)
    
Dès lors que l'image est transformée, on peut donc chercher les contours des formes

    (contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
La liste des contours étant maintenant en notre possession nous pouvons donc commencer à faire notre fonction de reconnaissance. Pour cela nous allons utiliser la fonction d'approximation de polynômes de la librairie d'opencv.

    def detectshape(c):
      peri = cv2.arcLength(c, True)
      vertices = cv2.approxPolyDP(c, 0.04 * peri, True)
      sides = len(vertices)
      
Cette fonction nous permet donc de savoir le nombre de côtés que dispose chaque contour en utilisant la fonction "len". Ainsi une fois le nombre de côtés connu il ne reste plus qu'à faire plusieurs cas en fonction du nombre de côtés.

    if sides == 3:
        shape = 'triangle'
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(c)
        aspectratio = float(w) / h
        if aspectratio == 1:
            shape = 'carré'
        else:
            shape = "rectangle"
    elif sides == 5:
        shape = 'pentagone'
    elif sides == 6:
        shape = 'hexagone'
    elif sides == 8:
        shape = 'octogone'
    elif sides == 10:
        shape = 'étoile'
    else:
        shape = 'cercle'
    return shape
    
## Reconnaissance de forme de couleur

La dernière étape consiste à mettre en commun les deux étapes précédentes. 
La difficulté est que l'on ne peut pas faire les étapes l'une après l'autre, pour être plus clair, la première pensée serait de faire la reconnaissance de couleur (respectivement de forme) puis à partir du masque obtenus faire la reconnaissance de forme (resp. de couleur) mais les masques obtenu afin de reconnaitre les couleurs ou les formes sont inutilisables pour faire l'autre reconnaissance comme nous pouvons le voir avec la première version du programme (Track.py) qui détecte n'importe quelle forme de la couleur définie ce qui s'observe bien avec les vidéos V1.mov et V2.mov.
Résultat, il faut donc faire les 2 masques à partir de l'image de départ et mettre en commun les 2 listes de contours obtenues. Pour effectuer cette mise en commun, j'ai opté pour la solution qui est de faire une boucle "for" afin de travailler directement sur les contours présents dans les 2 listes et ne pas passer par une troisième liste avant de travailler sur celle-ci.

      for cg in contours and cntg:
        cnts.append(cg)
        shape = detectshape(cg)
        print_rectangle(img, shape, cg)
            
## Affichage des forme de couleur à l'écran

Afin de vérifier le bon fonctionnement de ce programme, il était nécessaire d'afficher ce que détectait le programme. Pour cela, il fallait afficher les contours que détectait le programme avec la fonction "print_rectangle" (qui dans ce cas n'affiche que les rectangles)

    def print_rectangle(img, shape, c):
    if shape == 'rectangle':
        moment = cv2.moments(c)
        cx = 0
        cy = 0
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
        cv2.drawContours(img, [c], -1, (125, 0, 0), 2)
        cv2.putText(img, 'rectangle vert', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 125), 2)

Dans cette fonction nous pouvons apercevoir des variables cx et cy. Ces variables désignent le centre de chaque objet. Ainsi nous avons la position de l'objet détecté dans l'image ce qui est très pratique pour effectuer un suivi d'objet.  

## Conclusion

Afin de conclure sur ce programme, celui-ci est capable de reconnaitre différentes formes de différentes couleurs tout en connaissant la position de l'objet dans l'image. Mais ce programme a des limites, en effet, selon l'intensité lumineuse présente dans la pièce, l'orientation de la lumière vis-à-vis de la direction de la webcam, il peut y avoir un contrejour ou un reflet sur l'objet ce qui change donc la perception de la couleur comme nous pouvons le voir avec les vidéos Contrejour_V1.mov et Contrejour_V2.mov en comparaison avec les vidéos V1.mov et V2.mov. La deuxième limite concerne la reconnaissance de forme puisque les contours étant définis à partir d'une approximation il se peut qu'un rectangle soit reconnu comme un pentagone si un des coins n'est pas reconnu comme à angle droit et donc la présence d'un cinquième coté aussi petit soit-il.

## Utilisation des différents codes

Lancement du code principal à partir d'une vidéo

    (python) Track.py -v ~/Python/test_video.mp4
    
Lancement du code principal à partir de la webcam

    (python) Track.py 
    
Lancement du programme permettant de définir l'intervalle de couleur en HSV à partir de la webcam

     (python) range_detector.py -w --filter HSV --preview
     
Lancement du programme permettant de définir l'intervalle de couleur en RGB à partir d'une image

    (python) range-detector.py --filter RGB --image /path/to/image.png

    
