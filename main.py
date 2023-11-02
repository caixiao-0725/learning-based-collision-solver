import pygame
from collide_detection import check_collision



if __name__ == '__main__':

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    # 设置窗口大小

    # 设置窗口标题
    pygame.display.set_caption("My Game")

    triangle1 = [(100, 100), (200, 100), (150, 200)]
    triangle2 = [(100, 150), (350, 150), (300, 250)]

    running = True
    while running:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.draw.polygon(screen, (255, 0, 0), triangle1)
        pygame.draw.polygon(screen, (0, 0, 255), triangle2)

        if check_collision(triangle1, triangle2):
            print("Collision detected!")

        pygame.display.flip()

    pygame.quit()