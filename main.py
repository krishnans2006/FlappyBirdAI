import os
import random

import neat
import pygame

pygame.init()
pygame.font.init()

WIN_WIDTH = 550
WIN_HEIGHT = 800

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bg.png")))


class Bird:
    IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))),
            pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))),
            pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]
    MAX_ROTATION = 25
    ROTATION_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.image_count = 0
        self.image = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2
        if d >= 16:
            d = 16
        if d < 0:
            d -= 2
        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VEL

    def draw(self, win):
        self.image_count += 1
        if self.image_count < self.ANIMATION_TIME:
            self.image = self.IMGS[0]
        elif self.image_count < self.ANIMATION_TIME * 2:
            self.image = self.IMGS[1]
        elif self.image_count < self.ANIMATION_TIME * 3:
            self.image = self.IMGS[2]
        elif self.image_count < self.ANIMATION_TIME * 4:
            self.image = self.IMGS[1]
        elif self.image_count == self.ANIMATION_TIME * 4 + 1:
            self.image = self.IMGS[0]
            self.image_count = 0

        if self.tilt <= -80:
            self.image = self.IMGS[1]
            self.image_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        top_point = bird_mask.overlap(top_mask, top_offset)
        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)

        if top_point or bottom_point:
            return True
        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMAGE = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMAGE, (self.x1, self.y))
        win.blit(self.IMAGE, (self.x2, self.y))


def draw_screen(win, birds, pipes, base, score):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    text = pygame.font.SysFont("comicsans", 50).render(f"Score: {score}", 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    pygame.display.update()


def main(genomes, config):
    networks = []
    gens = []
    birds = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        gens.append(genome)

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    score = 0
    base = Base(730)
    pipes = [Pipe(600)]
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            gens[x].fitness += 0.1
            output = networks[x].activate(
                (730 - bird.y, abs(bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))
            if output[0] >= 0.5:
                bird.jump()

        rem = []
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    gens[x].fitness -= 1
                    birds.pop(x)
                    networks.pop(x)
                    gens.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
                    score += 1

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            for genome in gens:
                genome.fitness += 5
            pipes.append(Pipe(600))

        for pipe in rem:
            pipes.remove(pipe)

        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() > 730 or bird.y < 0:
                birds.pop(x)
                networks.pop(x)
                gens.pop(x)

        if score > 80:
            break

        base.move()
        draw_screen(win, birds, pipes, base, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
