-- SQL Schema for the Hybrid Movie Recommender Project
-- This script creates the necessary tables with appropriate relationships.

-- Drop tables if they exist to start fresh.
DROP TABLE IF EXISTS `recommendations`;
DROP TABLE IF EXISTS `requests`;
DROP TABLE IF EXISTS `reviews`;
DROP TABLE IF EXISTS `films`;
DROP TABLE IF EXISTS `clients`;
DROP TABLE IF EXISTS `types`;

-- Table for film genres/types (e.g., Action, Comedy, Drama)
CREATE TABLE `types` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `name` VARCHAR(100) NOT NULL UNIQUE,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for clients/users
CREATE TABLE `clients` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `name` VARCHAR(255) NOT NULL,
  `age` INT,
  `gender` VARCHAR(50),
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for films/movies
CREATE TABLE `films` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `name` VARCHAR(255) NOT NULL,
  `details` TEXT,
  `language` VARCHAR(100),
  `type_id` INT,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (`type_id`) REFERENCES `types`(`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for film reviews
CREATE TABLE `reviews` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `film_id` INT NOT NULL,
  `client_id` INT NOT NULL,
  `rate` DECIMAL(3, 1) NOT NULL, -- Allows ratings like 4.5
  `content` TEXT,
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (`film_id`) REFERENCES `films`(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`client_id`) REFERENCES `clients`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for recommendation requests from clients
-- Note: Corrected table name from "requestes" to "requests" for standard English.
CREATE TABLE `requests` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `client_id` INT NOT NULL,
  `status` INT NOT NULL DEFAULT 0, -- 0: pending, 1: processed, -1: error
  `age` INT,
  `gender` VARCHAR(50),
  `city_id` INT, -- Kept for compatibility with your code
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (`client_id`) REFERENCES `clients`(`id`) ON DELETE CASCADE,
  INDEX `status_index` (`status`) -- Index on status for faster querying of pending requests
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table to store the generated recommendations for each request
CREATE TABLE `recommendations` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `request_id` INT NOT NULL,
  `film_id` INT NOT NULL,
  `priority` INT NOT NULL, -- Lower number means higher priority
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (`request_id`) REFERENCES `requests`(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`film_id`) REFERENCES `films`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;