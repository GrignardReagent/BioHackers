'use client';

import { Button, Checkbox, FileInput, Grid, Image, SimpleGrid, Stack } from "@mantine/core";

export default function Home() {

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);

    fetch('//localhost:8000/upload', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        console.log('Success:', data);
        window.location.href = '/results';
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }

  return (
    <>
      <form className="m-3" onSubmit={handleSubmit}>
        <FileInput
          m={3}
          placeholder="Upload image"
          name="image"
        />
        <FileInput
          m={3}
          placeholder="Upload red image"
          name="redImage"
        />
        <FileInput
          m={3}
          placeholder="Upload lidar image"
          name="lidarImage"
        />
        <Button m={3} type="submit">Upload</Button>
      </form>
    </>
  );
}
