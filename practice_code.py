import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import random

# Фильтрация изображения
def filter_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.bilateralFilter(gray,6,75,75)
    edges = cv2.Canny(blurred,0,150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 
    return edges

# Класс прямоульника
class rectangle:
    def __init__(self, x, y, w, h, theta):
        thresh = 4
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.theta = theta
        s = np.sin(theta)
        c = np.cos(theta)
        centerx = x + w/2
        centery = y + h/2
        self.x1 = x
        self.y1 = y+h
        self.x2 = x+w
        self.y2 = y+h 
        self.x3 = x+w
        self.y3 = y
        x0r = centerx + (self.x - centerx)*c - (self.y - centery)*s
        y0r = centery + (self.y - centery)*c + (self.x - centerx)*s
        x1r = centerx + (self.x1 - centerx)*c - (self.y1 - centery)*s
        y1r = centery + (self.y1 - centery)*c + (self.x1 - centerx)*s
        x2r = centerx + (self.x2 - centerx)*c - (self.y2 - centery)*s
        y2r = centery + (self.y2 - centery)*c + (self.x2 - centerx)*s
        x3r = centerx + (self.x3 - centerx)*c - (self.y3 - centery)*s
        y3r = centery + (self.y3 - centery)*c + (self.x3 - centerx)*s
        self.pts = np.array([[x0r,y0r],[x1r,y1r],[x2r,y2r],[x3r,y3r]], np.int32)
        
        # Внешний четырехугольник

        out_x0 = self.x - thresh 
        out_y0 = self.y - thresh
        out_x1 = self.x1 - thresh
        out_y1 = self.y1 + thresh
        out_x2 = self.x2 + thresh
        out_y2 = self.y2 + thresh
        out_x3 = self.x3 + thresh
        out_y3 = self.y3 - thresh

        x0r = centerx + (out_x0 - centerx)*c - (out_y0 - centery)*s
        y0r = centery + (out_y0 - centery)*c + (out_x0 - centerx)*s
        x1r = centerx + (out_x1 - centerx)*c - (out_y1 - centery)*s
        y1r = centery + (out_y1 - centery)*c + (out_x1 - centerx)*s
        x2r = centerx + (out_x2 - centerx)*c - (out_y2 - centery)*s
        y2r = centery + (out_y2 - centery)*c + (out_x2 - centerx)*s
        x3r = centerx + (out_x3 - centerx)*c - (out_y3 - centery)*s
        y3r = centery + (out_y3 - centery)*c + (out_x3 - centerx)*s
        self.pts_outer = np.array([[x0r,y0r],[x1r,y1r],[x2r,y2r],[x3r,y3r]], np.int32)

        # Внутренний четырехугольник

        int_x0 = self.x + thresh
        int_y0 = self.y + thresh
        int_x1 = self.x1 + thresh
        int_y1 = self.y1 - thresh
        int_x2 = self.x2 - thresh
        int_y2 = self.y2 - thresh
        int_x3 = self.x3 - thresh
        int_y3 = self.y3 + thresh

        x0r = centerx + (int_x0 - centerx)*c - (int_y0 - centery)*s
        y0r = centery + (int_y0 - centery)*c + (int_x0 - centerx)*s
        x1r = centerx + (int_x1 - centerx)*c - (int_y1 - centery)*s
        y1r = centery + (int_y1 - centery)*c + (int_x1 - centerx)*s
        x2r = centerx + (int_x2 - centerx)*c - (int_y2 - centery)*s
        y2r = centery + (int_y2 - centery)*c + (int_x2 - centerx)*s
        x3r = centerx + (int_x3 - centerx)*c - (int_y3 - centery)*s
        y3r = centery + (int_y3 - centery)*c + (int_x3 - centerx)*s
        self.pts_inner = np.array([[x0r,y0r],[x1r,y1r],[x2r,y2r],[x3r,y3r]], np.int32)
        
        self.S_thresh = cv2.contourArea(self.pts_outer) - cv2.contourArea(self.pts_inner) 
        self.num_whites = 0


# Класс генетического алгоритма
class GA:
    def __init__(self, image):
        self.population=[]
        self.image = image
        self.img_dims = image.shape[:2]
        self.fitness =[]
        self.num_whites_total = self.image[self.image==(255,255,255)].size
        print(self.num_whites_total)
        
    def init_popultation(self, size):
        self.size = size
        for _ in range(self.size):
            rand_x = np.random.randint(0, self.img_dims[0]-10,1)
            rand_y = np.random.randint(0, self.img_dims[1]-10,1)
            rand_w = np.random.randint(10, self.img_dims[0]*0.9,1)
            rand_h = np.random.randint(10, self.img_dims[1]*0.9,1)
            rand_theta = np.random.uniform(0, 2*np.pi)
            self.population.append(rectangle(rand_x,rand_y,rand_w,rand_h,rand_theta))
            
    def calc_axis_line(self, point_1, point_2):
        k = (point_1[1] - point_2[1])/(point_1[0]-point_2[0]+ sys.float_info.epsilon)
        b = point_2[1]-k*point_2[0]
        return k, b
    
    def find_pxls_thresh(self):
        for i in range(self.size):
            mask_img = np.zeros(self.image.shape[:2], np.uint8)
            cv2.fillPoly(mask_img, pts = [self.population[i].pts_outer], color=(255,255,255))
            cv2.fillPoly(mask_img, pts = [self.population[i].pts_inner], color=(0,0,0))
            masked = cv2.bitwise_and(self.image, self.image, mask=mask_img)
            self.population[i].num_whites = masked[masked==(255,255,255)].size
        
    def calc_fitness(self):
        self.fitness =[]
        weighted_sides = []
        for i in range(self.size):  
            self.fitness.append( 100*(self.population[i].num_whites/(self.num_whites_total-self.population[i].num_whites+1)))
        x = zip(self.fitness,self.population)
        xs = sorted(x,reverse = True, key=lambda tup: tup[0])
        self.fitness = [x[0] for x in xs]
        self.population = [x[1] for x in xs]
    
    def selection(self):
        self.population = self.population[:int(len(self.population)/2)]
        
    def crossover(self, r1, r2):
        a = np.random.random(1)
        b = 1 - a
        x_new_1 = a * r1.x + b * r2.x
        y_new_1 = a * r1.y + b * r2.y
        w_new_1 = a * r1.w + b * r2.w
        h_new_1 = a * r1.h + b * r2.h
        theta_new_1 = a * r1.theta + b * r2.theta
        self.rect_new_1 = rectangle(x_new_1,y_new_1,w_new_1,h_new_1,theta_new_1)
        a = 1 - a
        b = 1 - b
        x_new_2 = a * r1.x + b * r2.x
        y_new_2 = a * r1.y + b * r2.y
        w_new_2 = a * r1.w + b * r2.w
        h_new_2 = a * r1.h + b * r2.h
        theta_new_2 = a * r1.theta + b * r2.theta
        self.rect_new_2 = rectangle(x_new_2,y_new_2,w_new_2,h_new_2,theta_new_2)
        return self.rect_new_1, self.rect_new_2
        
    def repopulate(self):
        self.offspring = []
        for _ in range(len(self.population)):
            index_1 = np.random.randint(0,len(self.population)-1,1)
            index_2 = np.random.randint(index_1,len(self.population)-1,1)
            child_1, child_2 = self.crossover(self.population[index_1[0]],self.population[index_2[0]]) 
            self.offspring.append(child_1)
        self.population = self.population + self.offspring
        
    def mutate(self, rect):
        rect.x += int(10*(1-2*np.random.random(1)[0]))
        rect.y += int(10*(1-2*np.random.random(1)[0]))
        rect.w += int(10*(1-2*np.random.random(1)[0]))
        rect.h += int(10*(1-2*np.random.random(1)[0]))
        rect.theta += 10*(1-2*np.random.random(1)[0])
        return rect
    
    def mutate_population(self):
        for i in range(len(self.population)):
            self.population[i] = self.mutate(self.population[i])


if __name__ == "__main__":
    #np.random.seed(14)
    file = cv2.imread('Test/box1.jpg')
    scale_percent = 70 
    width = int(file.shape[1] * scale_percent / 100)
    height = int(file.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(file, dim)
    edges = filter_image(resized)
    edges_cpy = edges.copy()
    population_len =30 
    algorithm = GA(edges_cpy)
    number_steps =500
    algorithm.init_popultation(population_len )
    best_fit = 0
    best_model = None
    for _ in range(number_steps):
        edges_cpy = edges.copy()
        random.shuffle(algorithm.population)
        for i in range(population_len):
            cv2.polylines(edges_cpy,[algorithm.population[i].pts],True,(0,255,255))
            cv2.polylines(edges_cpy,[algorithm.population[i].pts_outer],True,(0,0,255))
            cv2.polylines(edges_cpy,[algorithm.population[i].pts_inner],True,(0,0,255))
        algorithm.find_pxls_thresh()
        algorithm.calc_fitness()
        if algorithm.fitness[0]>best_fit:
            best_fit = algorithm.fitness[0]
            print('Found better fintess: ',best_fit )
            best_model = algorithm.population[0]
        algorithm.selection()
        algorithm.repopulate()
        algorithm.mutate_population()
        cv2.imshow('edges', edges_cpy)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    edges_cpy = edges.copy()
    print('Best fintess: ',best_fit )
    cv2.polylines(edges_cpy,[best_model.pts],True,(0,255,255))
    cv2.polylines(edges_cpy,[best_model.pts_outer],True,(0,0,255))
    cv2.polylines(edges_cpy,[best_model.pts_inner],True,(0,0,255))
    cv2.imshow('edges', edges_cpy)
    cv2.waitKey()
    cv2.destroyAllWindows()