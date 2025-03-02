'use client';

import { Checkbox, FileInput, Grid, Image, SimpleGrid, Stack } from "@mantine/core";

export default function Home() {
  return (
    <>
      <div className="m-3">
        <FileInput
          placeholder="Upload image"
        />
      </div>
      <div className="m-3">
        <SimpleGrid cols={2}>
          <div>
            <Image
              src="https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png"
            />
          </div>
          <div>
            <Stack>
              <Checkbox label="Land use" />
            </Stack>
          </div>
        </SimpleGrid>
      </div>
    </>
  );
}
